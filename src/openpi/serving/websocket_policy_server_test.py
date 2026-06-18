import asyncio
from collections.abc import AsyncIterator
import contextlib
import threading
from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import pytest
import websockets
import websockets.asyncio.client as _client
import websockets.asyncio.server as _server

from openpi.serving import websocket_policy_server as _websocket_policy_server


class _ImmediatePolicy(_base_policy.BasePolicy):
    def infer(self, obs: dict) -> dict:
        return {"request": obs["request"]}


class _ErrorPolicy(_base_policy.BasePolicy):
    def infer(self, obs: dict) -> dict:
        raise RuntimeError(f"boom {obs['request']}")


class _BlockingPolicy(_base_policy.BasePolicy):
    def __init__(self, loop: asyncio.AbstractEventLoop, release_event: threading.Event) -> None:
        self._loop = loop
        self._release_event = release_event
        self._lock = threading.Lock()
        self._started_queue: asyncio.Queue[int] = asyncio.Queue()
        self._active_count = 0
        self._started_count = 0
        self._max_active_count = 0

    def infer(self, obs: dict) -> dict:
        with self._lock:
            self._active_count += 1
            self._started_count += 1
            started_count = self._started_count
            self._max_active_count = max(self._max_active_count, self._active_count)

        self._loop.call_soon_threadsafe(self._started_queue.put_nowait, started_count)
        try:
            if not self._release_event.wait(timeout=5):
                raise TimeoutError("Timed out waiting for test to release fake policy")
            return {"request": obs["request"]}
        finally:
            with self._lock:
                self._active_count -= 1

    @property
    def max_active_count(self) -> int:
        with self._lock:
            return self._max_active_count

    async def wait_started(self, *, timeout: float = 1.0) -> int:
        return await asyncio.wait_for(self._started_queue.get(), timeout=timeout)

    async def assert_no_new_start(self) -> None:
        with pytest.raises(asyncio.TimeoutError):
            await self.wait_started(timeout=0.05)


@contextlib.asynccontextmanager
async def _serve_policy(
    policy: _base_policy.BasePolicy,
    *,
    inference_workers: int = 1,
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[tuple[_websocket_policy_server.WebsocketPolicyServer, str]]:
    policy_server = _websocket_policy_server.WebsocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=0,
        metadata=metadata or {"test": True},
        inference_workers=inference_workers,
    )
    async with _server.serve(
        policy_server._handler,  # noqa: SLF001
        "127.0.0.1",
        0,
        compression=None,
        max_size=None,
        process_request=_websocket_policy_server._health_check,  # noqa: SLF001
    ) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        try:
            yield policy_server, f"ws://127.0.0.1:{port}"
        finally:
            policy_server.close()


async def _connect(uri: str) -> tuple[_client.ClientConnection, dict]:
    websocket = await _client.connect(uri, compression=None, max_size=None)
    metadata = msgpack_numpy.unpackb(await websocket.recv())
    return websocket, metadata


async def _send_request(websocket: _client.ClientConnection, request: int) -> None:
    await websocket.send(msgpack_numpy.packb({"request": request}))


async def _recv_response(websocket: _client.ClientConnection) -> dict:
    response = await websocket.recv()
    assert not isinstance(response, str)
    return msgpack_numpy.unpackb(response)


async def _close_websockets(*websockets_to_close: _client.ClientConnection) -> None:
    for websocket in websockets_to_close:
        await websocket.close()


def test_metadata_handshake_stays_responsive_during_blocked_inference() -> None:
    async def run_test() -> None:
        release_event = threading.Event()
        policy = _BlockingPolicy(asyncio.get_running_loop(), release_event)

        async with _serve_policy(policy, inference_workers=1, metadata={"server": "test"}) as (_, uri):
            websocket_a, metadata_a = await _connect(uri)
            websocket_b = None
            try:
                assert metadata_a == {"server": "test"}
                await _send_request(websocket_a, 1)
                assert await policy.wait_started() == 1

                websocket_b, metadata_b = await asyncio.wait_for(_connect(uri), timeout=0.5)
                assert metadata_b == {"server": "test"}

                release_event.set()
                response_a = await asyncio.wait_for(_recv_response(websocket_a), timeout=1)
                assert response_a["request"] == 1
            finally:
                release_event.set()
                await _close_websockets(websocket_a, *([websocket_b] if websocket_b is not None else []))

    asyncio.run(run_test())


def test_inference_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="inference_workers must be >= 1"):
        _websocket_policy_server.WebsocketPolicyServer(_ImmediatePolicy(), inference_workers=0)


def test_default_worker_count_serializes_policy_calls() -> None:
    async def run_test() -> None:
        release_event = threading.Event()
        policy = _BlockingPolicy(asyncio.get_running_loop(), release_event)

        async with _serve_policy(policy, inference_workers=1) as (_, uri):
            websocket_a, _ = await _connect(uri)
            websocket_b, _ = await _connect(uri)
            try:
                await _send_request(websocket_a, 1)
                await _send_request(websocket_b, 2)

                assert await policy.wait_started() == 1
                await policy.assert_no_new_start()

                release_event.set()
                responses = await asyncio.gather(_recv_response(websocket_a), _recv_response(websocket_b))
                assert sorted(response["request"] for response in responses) == [1, 2]
                assert policy.max_active_count == 1
            finally:
                release_event.set()
                await _close_websockets(websocket_a, websocket_b)

    asyncio.run(run_test())


def test_configured_worker_limit_is_respected() -> None:
    async def run_test() -> None:
        release_event = threading.Event()
        policy = _BlockingPolicy(asyncio.get_running_loop(), release_event)

        async with _serve_policy(policy, inference_workers=2) as (_, uri):
            websockets_to_close = []
            try:
                for request in range(3):
                    websocket, _ = await _connect(uri)
                    websockets_to_close.append(websocket)
                    await _send_request(websocket, request)

                assert [await policy.wait_started(), await policy.wait_started()] == [1, 2]
                await policy.assert_no_new_start()

                release_event.set()
                responses = await asyncio.gather(*[_recv_response(websocket) for websocket in websockets_to_close])
                assert sorted(response["request"] for response in responses) == [0, 1, 2]
                assert policy.max_active_count == 2
            finally:
                release_event.set()
                await _close_websockets(*websockets_to_close)

    asyncio.run(run_test())


def test_inference_errors_still_send_traceback_string_then_close() -> None:
    async def run_test() -> None:
        async with _serve_policy(_ErrorPolicy()) as (_, uri):
            websocket, _ = await _connect(uri)
            try:
                await _send_request(websocket, 7)
                response = await websocket.recv()
                assert isinstance(response, str)
                assert "RuntimeError: boom 7" in response

                with pytest.raises(websockets.ConnectionClosed):
                    await websocket.recv()
            finally:
                await websocket.close()

    asyncio.run(run_test())


def test_successful_responses_include_server_timing() -> None:
    async def run_test() -> None:
        async with _serve_policy(_ImmediatePolicy()) as (_, uri):
            websocket, _ = await _connect(uri)
            try:
                await _send_request(websocket, 3)
                response = await _recv_response(websocket)
                assert response["request"] == 3
                assert response["server_timing"]["infer_ms"] >= 0
            finally:
                await websocket.close()

    asyncio.run(run_test())


def test_cancelled_async_inference_keeps_worker_slot_until_thread_finishes() -> None:
    async def run_test() -> None:
        release_event = threading.Event()
        policy = _BlockingPolicy(asyncio.get_running_loop(), release_event)
        policy_server = _websocket_policy_server.WebsocketPolicyServer(policy, inference_workers=1)
        try:
            first_task = asyncio.create_task(policy_server._infer_async({"request": 1}))  # noqa: SLF001
            assert await policy.wait_started() == 1

            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task

            second_task = asyncio.create_task(policy_server._infer_async({"request": 2}))  # noqa: SLF001
            await policy.assert_no_new_start()

            release_event.set()
            response = await asyncio.wait_for(second_task, timeout=1)
            assert response["request"] == 2
            assert policy.max_active_count == 1
        finally:
            release_event.set()
            policy_server.close()

    asyncio.run(run_test())
