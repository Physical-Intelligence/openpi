# ruff: noqa: RUF002, RUF003
"""基于主机单调时钟的多相机最近帧同步器。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from itertools import product
import threading
import time

from examples.ur5_twinwrist.cameras.models import CameraFrame


class FrameSynchronizer:
    """缓存每路最近帧并返回时间最接近的最新完整组合。

    D435 的设备时钟彼此独立，因此跨相机同步只使用同一进程里的
    ``time.monotonic_ns``。原始设备时间戳仍原样保留供数据审计。
    """

    def __init__(
        self,
        roles: tuple[str, ...],
        *,
        queue_size: int = 8,
        max_skew_ms: float = 50.0,
        max_frame_age_ms: float = 250.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if len(roles) < 2 or len(set(roles)) != len(roles):
            raise ValueError("synchronizer requires at least two unique camera roles")
        if queue_size < 2:
            raise ValueError("queue_size must be at least 2")
        if max_skew_ms <= 0.0 or max_frame_age_ms <= 0.0:
            raise ValueError("skew and age limits must be positive")
        self._roles = roles
        self._queues = {role: deque(maxlen=queue_size) for role in roles}
        self._max_skew_ns = round(max_skew_ms * 1_000_000.0)
        self._max_frame_age_ns = round(max_frame_age_ms * 1_000_000.0)
        self._clock_ns = clock_ns
        self._condition = threading.Condition()
        self._failures: dict[str, BaseException] = {}
        self._closed = False

    def add(self, frame: CameraFrame) -> None:
        """添加一帧并唤醒等待读取者。"""

        if frame.role not in self._queues:
            raise ValueError(f"unknown camera role: {frame.role}")
        with self._condition:
            if self._closed:
                return
            self._queues[frame.role].append(frame)
            self._condition.notify_all()

    def report_failure(self, role: str, error: BaseException) -> None:
        """让后台采集错误立即穿透到阻塞中的 ``read``。"""

        with self._condition:
            self._failures.setdefault(role, error)
            self._condition.notify_all()

    def close(self) -> None:
        """唤醒读取线程并阻止继续加入帧。"""

        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def raise_if_failed(self) -> None:
        """若任一相机线程已经失败，立即抛出带 role 的异常。"""

        with self._condition:
            self._raise_failure()

    def read(self, timeout_s: float) -> dict[str, CameraFrame]:
        """读取满足 skew/age 门禁的一组三帧，超时则 fail closed。"""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        # 等待期限使用独立的真实 monotonic 计时；注入的 clock_ns 只描述帧
        # 时间轴，测试或设备层即使暂时不推进该时钟也不能造成永久阻塞。
        deadline_s = time.monotonic() + timeout_s
        with self._condition:
            while True:
                self._raise_failure()
                if self._closed:
                    raise RuntimeError("camera synchronizer is closed")
                chosen = self._choose_frames(self._clock_ns()) if all(self._queues.values()) else None
                if chosen is not None:
                    self._consume_through(chosen)
                    return chosen

                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(self._timeout_message())
                self._condition.wait(remaining_s)

    def _raise_failure(self) -> None:
        if not self._failures:
            return
        role, error = next(iter(self._failures.items()))
        raise RuntimeError(f"camera {role!r} capture failed: {error}") from error

    def _choose_frames(self, now_ns: int) -> dict[str, CameraFrame] | None:
        feasible: list[tuple[int, int, tuple[CameraFrame, ...]]] = []
        for frames in product(*(self._queues[role] for role in self._roles)):
            timestamps = [frame.host_timestamp_ns for frame in frames]
            skew_ns = max(timestamps) - min(timestamps)
            age_ns = now_ns - min(timestamps)
            if skew_ns <= self._max_skew_ns and 0 <= age_ns <= self._max_frame_age_ns:
                # 优先选择最新完整组合；时间相同时再选 skew 更小的组合。
                feasible.append((min(timestamps), -skew_ns, frames))
        if not feasible:
            return None
        frames = max(feasible, key=lambda item: item[:2])[2]
        return dict(zip(self._roles, frames, strict=True))

    def _consume_through(self, chosen: dict[str, CameraFrame]) -> None:
        # 每个 chosen frame 及其旧帧只消费一次，防止记录器重复保存同一帧。
        for role, selected in chosen.items():
            queue = self._queues[role]
            while queue:
                frame = queue.popleft()
                if frame is selected:
                    break

    def _timeout_message(self) -> str:
        now_ns = self._clock_ns()
        latest = {
            role: (
                None
                if not queue
                else {
                    "sequence": queue[-1].sequence,
                    "host_timestamp_ns": queue[-1].host_timestamp_ns,
                    "age_ms": (now_ns - queue[-1].host_timestamp_ns) / 1_000_000.0,
                }
            )
            for role, queue in self._queues.items()
        }
        available = [item["host_timestamp_ns"] for item in latest.values() if item is not None]
        latest_skew_ms = (max(available) - min(available)) / 1_000_000.0 if len(available) == len(self._roles) else None
        return (
            "timed out waiting for synchronized camera frames; "
            f"max_skew_ms={self._max_skew_ns / 1_000_000.0} "
            f"latest_skew_ms={latest_skew_ms!r} latest={latest}"
        )
