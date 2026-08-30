"""Single-owner worker and cached state facade for legacy serial grippers.

The legacy Hiwonder and Feetech drivers protect each transaction with a lock,
but the collection control loop and synchronizer both call ``read_state``.
That still gives two application threads ownership of one serial device.  This
facade moves every operation which can touch the serial bus to one worker
thread.  Callers only see a cached state or submit a synchronous command.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import queue
import threading
import time
from typing import Any


@dataclass
class _Request:
    operation: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


_STOP = object()


class SingleOwnerGripper:
    """Expose the legacy gripper contract while one thread owns the device.

    ``open``, all reads and writes, and ``close`` on the delegate execute in
    the same worker thread.  A command waits for both the write and an
    immediate safety-enforced feedback read.  Any delegate exception is
    propagated to the caller and permanently fails the facade closed.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        poll_interval_s: float = 0.05,
        stale_timeout_s: float = 0.5,
        operation_timeout_s: float = 2.0,
        open_timeout_s: float = 2.0,
        close_timeout_s: float = 2.0,
    ) -> None:
        if poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        if stale_timeout_s <= poll_interval_s:
            raise ValueError("stale_timeout_s must exceed poll_interval_s")
        for name, value in (
            ("operation_timeout_s", operation_timeout_s),
            ("open_timeout_s", open_timeout_s),
            ("close_timeout_s", close_timeout_s),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        self._delegate = delegate
        self._poll_interval_s = float(poll_interval_s)
        self._stale_timeout_s = float(stale_timeout_s)
        self._operation_timeout_s = float(operation_timeout_s)
        self._open_timeout_s = float(open_timeout_s)
        self._close_timeout_s = float(close_timeout_s)

        self._requests: queue.Queue[_Request | object] = queue.Queue()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._latest_state: Any = None
        self._latest_state_at_s: float | None = None
        self._opened = False
        self._closed = False

    @property
    def driver_name(self) -> str:
        return str(self._delegate.driver_name)

    @property
    def port(self) -> str:
        return str(self._delegate.port)

    @property
    def servo_id(self) -> int:
        return int(self._delegate.servo_id)

    @property
    def open_position(self) -> int:
        return int(self._delegate.open_position)

    @property
    def closed_position(self) -> int:
        return int(self._delegate.closed_position)

    def __enter__(self) -> SingleOwnerGripper:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def open(self) -> None:
        """Start the owner thread and open the delegate inside that thread."""
        with self._lifecycle_lock:
            if self._opened:
                raise RuntimeError("gripper worker is already open")
            if self._closed:
                raise RuntimeError("gripper worker cannot be reopened")
            self._thread = threading.Thread(
                target=self._run,
                name="ur5-twinwrist-gripper-owner",
                daemon=True,
            )
            self._thread.start()
        if not self._ready.wait(self._open_timeout_s):
            failure = TimeoutError("gripper worker did not open before timeout")
            self._record_failure(failure)
            self._request_stop()
            raise failure
        self._raise_failure()
        with self._lifecycle_lock:
            self._opened = True

    def close(self) -> None:
        """Stop the worker; the worker itself closes the delegate."""
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                self._closed = True
                return
            self._opened = False
            self._closed = True
        self._request_stop()
        thread.join(self._close_timeout_s)
        if thread.is_alive():
            failure = TimeoutError("gripper worker did not close before timeout")
            self._record_failure(failure)
            raise failure
        self._raise_failure()

    def command(self, value: int) -> None:
        """Submit a binary 0=open/1=closed command to the owner."""
        self._submit("command", value)

    def command_position(self, value: float) -> None:
        """Submit an absolute normalized-position command to the owner."""
        self._submit("command_position", value)

    def unload(self) -> None:
        """Submit the driver's torque-disable operation to the owner."""
        self._submit("unload")

    def read_state(self, *, enforce_safety: bool = True) -> Any:
        """Return the last safety-enforced worker sample without bus access.

        ``enforce_safety=False`` is accepted for protocol compatibility, but
        cached samples are always collected with safety enabled.
        """
        del enforce_safety
        self._ensure_open()
        self._raise_failure()
        with self._state_lock:
            state = self._latest_state
            sampled_at = self._latest_state_at_s
        if state is None or sampled_at is None:
            raise RuntimeError("gripper worker has no cached state")
        age_s = time.monotonic() - sampled_at
        if age_s > self._stale_timeout_s:
            failure = TimeoutError(f"gripper cached state is stale: {age_s:.3f}s > {self._stale_timeout_s:.3f}s")
            self._record_failure(failure)
            self._request_stop()
            raise failure
        return state

    def healthy(self) -> bool:
        """Return whether the owner is alive and its feedback cache is fresh."""
        with self._failure_lock:
            if self._failure is not None:
                return False
        with self._state_lock:
            sampled_at = self._latest_state_at_s
        thread = self._thread
        return bool(
            self._opened
            and thread is not None
            and thread.is_alive()
            and sampled_at is not None
            and time.monotonic() - sampled_at <= self._stale_timeout_s
        )

    def _ensure_open(self) -> None:
        if not self._opened:
            self._raise_failure()
            raise RuntimeError("gripper worker is not open")

    def _submit(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        self._ensure_open()
        self._raise_failure()
        request = _Request(operation=operation, args=args, kwargs=kwargs)
        self._requests.put(request)
        if not request.done.wait(self._operation_timeout_s):
            failure = TimeoutError(f"gripper {operation} timed out")
            self._record_failure(failure)
            self._request_stop()
            raise failure
        if request.error is not None:
            raise request.error
        self._raise_failure()
        return request.result

    def _request_stop(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            self._requests.put(_STOP)

    def _run(self) -> None:
        try:
            self._delegate.open()
            self._refresh_state()
            self._ready.set()
            next_poll_s = time.monotonic() + self._poll_interval_s
            while not self._stop.is_set():
                timeout_s = max(0.0, next_poll_s - time.monotonic())
                try:
                    item = self._requests.get(timeout=timeout_s)
                except queue.Empty:
                    item = None
                if item is _STOP:
                    break
                if isinstance(item, _Request):
                    self._execute(item)
                if time.monotonic() >= next_poll_s:
                    self._refresh_state()
                    next_poll_s = time.monotonic() + self._poll_interval_s
        except BaseException as exc:
            self._record_failure(exc)
        finally:
            self._ready.set()
            try:
                self._delegate.close()
            except BaseException as exc:
                self._record_failure(exc)
            self._fail_pending_requests()

    def _execute(self, request: _Request) -> None:
        try:
            operation = getattr(self._delegate, request.operation)
            request.result = operation(*request.args, **request.kwargs)
            # A synchronous command is complete only after measured feedback
            # has been refreshed and its safety checks have passed.
            self._refresh_state()
        except BaseException as exc:
            request.error = exc
            self._record_failure(exc)
            self._stop.set()
            raise
        finally:
            request.done.set()

    def _refresh_state(self) -> None:
        state = self._delegate.read_state(enforce_safety=True)
        sampled_at = time.monotonic()
        with self._state_lock:
            self._latest_state = state
            self._latest_state_at_s = sampled_at

    def _record_failure(self, failure: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = failure

    def _raise_failure(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def _fail_pending_requests(self) -> None:
        with self._failure_lock:
            failure = self._failure or RuntimeError("gripper worker stopped")
        while True:
            try:
                item = self._requests.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _Request):
                item.error = failure
                item.done.set()
