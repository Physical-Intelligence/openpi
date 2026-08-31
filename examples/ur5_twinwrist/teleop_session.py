# ruff: noqa: RUF001, RUF002, RUF003
"""SpaceMouse 真机遥操作与 10 Hz HDF5 录制会话。

125 Hz 控制 keepalive 在调用线程运行；相机同步等待和 HDF5 写入固定在独立
recorder 线程中。所有依赖均可注入，单元测试不导入或连接真实设备。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import math
from pathlib import Path
from queue import Empty
from queue import Queue
import threading
import time
from typing import Any, Protocol

import numpy as np

from examples.ur5_twinwrist.controller.spacemouse import motion_to_ur5_twist
from examples.ur5_twinwrist.episode_recorder import EpisodeRecorder
from examples.ur5_twinwrist.teleop_controls import gripper_target_from_buttons
from examples.ur5_twinwrist.teleop_hardware import ActionReceipt
from examples.ur5_twinwrist.teleop_hardware import MaintenanceStatus
from examples.ur5_twinwrist.teleop_hardware import spacemouse_wrist_velocity_deg_s


class SessionError(RuntimeError):
    """遥操作会话不能安全继续。"""


class _RecorderOperation(str, Enum):
    START = "start"
    SAVE = "save"
    REJECT = "reject"
    CLOSE = "close"


@dataclass(slots=True)
class _RecorderRequest:
    operation: _RecorderOperation
    reason: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None

class _Mouse(Protocol):
    def connect(self) -> None: ...
    def read(self, *, fail_on_stale: bool = False) -> Any: ...
    def close(self) -> None: ...


class _Hardware(Protocol):
    def connect(self) -> None: ...
    def update_command(
        self,
        ur_speed_l: Any,
        *,
        wrist_velocity_deg_s: Any,
        gripper_target: float,
        issued_monotonic_ns: int | None = None,
    ) -> Any: ...
    def get_observation(self) -> dict[str, Any]: ...
    def latest_action_receipt(self) -> ActionReceipt: ...
    def set_episode_active(self, *, active: bool) -> None: ...
    def set_joint6_jog(self, direction: int) -> None: ...
    def request_home(self) -> None: ...
    def request_wrist_master_resume(self) -> None: ...
    def maintenance_status(self) -> MaintenanceStatus: ...
    def report_input_failure(self, failure: BaseException) -> None: ...
    def raise_if_failed(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


RecorderFactory = Callable[[str | Path, Mapping[str, Any]], Any]


class _RecorderThread:
    """唯一持有 EpisodeRecorder 的线程。"""

    def __init__(
        self,
        hardware: _Hardware,
        project: Mapping[str, Any],
        output: str | Path,
        *,
        recorder_factory: RecorderFactory,
        monotonic: Callable[[], float],
    ) -> None:
        self._hardware = hardware
        self._project = project
        self._output = output
        self._factory = recorder_factory
        self._monotonic = monotonic
        self._period_s = 1.0 / float(project["collection"]["capture"]["record_hz"])
        self._queue: Queue[_RecorderRequest] = Queue()
        self._lock = threading.Lock()
        self._failure: BaseException | None = None
        self._active = False
        self._thread = threading.Thread(target=self._run, name="ur5-twinwrist-hdf5-recorder", daemon=True)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def start(self) -> None:
        self._thread.start()

    def submit(self, operation: _RecorderOperation, *, reason: str = "") -> _RecorderRequest:
        self.raise_if_failed()
        request = _RecorderRequest(operation, reason)
        self._queue.put(request)
        return request

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise SessionError(f"录制线程已失败：{failure}") from failure

    def close(self, *, timeout_s: float = 10.0) -> None:
        if not self._thread.is_alive():
            return
        request = _RecorderRequest(_RecorderOperation.CLOSE, "会话关闭")
        self._queue.put(request)
        request.done.wait(timeout_s)
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("HDF5 recorder 线程未在期限内停止")

    def _run(self) -> None:
        recorder: Any | None = None
        current_request: _RecorderRequest | None = None
        next_capture = self._monotonic()
        try:
            recorder = self._factory(self._output, self._project)
            while True:
                timeout = max(0.0, next_capture - self._monotonic()) if self.active else 0.1
                try:
                    current_request = self._queue.get(timeout=timeout)
                except Empty:
                    current_request = None
                if current_request is not None:
                    if self._handle(recorder, current_request):
                        return
                    current_request.done.set()
                    current_request = None
                    if self.active:
                        next_capture = min(next_capture, self._monotonic())
                if self.active and self._monotonic() >= next_capture:
                    self._append_one(recorder)
                    next_capture = self._monotonic() + self._period_s
        except BaseException as exc:
            if current_request is not None:
                current_request.error = exc
                current_request.done.set()
            if recorder is not None and bool(getattr(recorder, "active", False)):
                with suppress(Exception):
                    recorder.reject(f"recorder thread failure: {type(exc).__name__}: {exc}")
            self._set_active(value=False)
            self._set_failure(exc)
            with suppress(Exception):
                self._hardware.report_input_failure(exc)
            self._fail_pending(exc)
        finally:
            if recorder is not None:
                with suppress(Exception):
                    recorder.close()

    def _handle(self, recorder: Any, request: _RecorderRequest) -> bool:
        if request.operation is _RecorderOperation.START:
            if self.active:
                raise RuntimeError("已有 episode 正在录制")
            observation = self._hardware.get_observation()
            request.result = recorder.start(observation)
            self._set_active(value=True)
            # 首帧也必须使用硬件实际成功下发的 receipt，而不是候选 action。
            recorder.append_if_due(observation, observation["action_receipt"].action)
            return False
        if request.operation is _RecorderOperation.SAVE:
            if not self.active:
                raise RuntimeError("没有活动 episode 可保存")
            request.result = recorder.save()
            self._set_active(value=False)
            return False
        if request.operation is _RecorderOperation.REJECT:
            if self.active:
                request.result = recorder.reject(request.reason or "operator rejected episode")
                self._set_active(value=False)
            return False
        if request.operation is _RecorderOperation.CLOSE:
            if self.active:
                request.result = recorder.reject(request.reason or "collector closed with active episode")
                self._set_active(value=False)
            request.done.set()
            return True
        raise AssertionError(f"未知 recorder operation: {request.operation}")

    def _append_one(self, recorder: Any) -> None:
        observation = self._hardware.get_observation()
        receipt = observation.get("action_receipt")
        if not isinstance(receipt, ActionReceipt) and not hasattr(receipt, "action"):
            raise RuntimeError("observation 缺少合法 action_receipt")
        recorder.append_if_due(observation, receipt.action)

    def _set_active(self, *, value: bool) -> None:
        with self._lock:
            self._active = value

    def _set_failure(self, failure: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = failure

    def _fail_pending(self, failure: BaseException) -> None:
        while True:
            try:
                request = self._queue.get_nowait()
            except Empty:
                return
            request.error = failure
            request.done.set()


class TeleopCollectionSession:
    """最小真机遥操作数采状态机。"""

    def __init__(
        self,
        project: Mapping[str, Any],
        *,
        episodes: int,
        spacemouse: _Mouse,
        hardware: _Hardware,
        output: str | Path,
        recorder_factory: RecorderFactory = EpisodeRecorder,
        console: Callable[[str], None] = print,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        home_timeout_s: float | None = None,
    ) -> None:
        if episodes < 1:
            raise ValueError("episodes 必须大于 0")
        resolved_home_timeout_s = float(
            project["poses"]["ur5"]["full_home_timeout_s"]
            if home_timeout_s is None
            else home_timeout_s
        )
        if not math.isfinite(resolved_home_timeout_s) or resolved_home_timeout_s <= 0.0:
            raise ValueError("home_timeout_s 必须是正有限数")
        self.project = project
        self.episodes = episodes
        self.mouse = spacemouse
        self.hardware = hardware
        self.output = output
        self._recorder_factory = recorder_factory
        self._console = console
        self._monotonic = monotonic
        self._sleep = sleeper
        self._home_timeout_s = resolved_home_timeout_s
        self._control_hz = float(project["hardware"]["ur5"]["control_hz"])
        self._control_period_s = 1.0 / self._control_hz
        self._gripper_target = float(project["poses"]["gripper"]["home_value"])
        self._recording = False
        self._previous_buttons: dict[str, bool] = {}
        self._recorder: _RecorderThread | None = None
        self._successful = 0

    def run(self) -> dict[str, Any]:
        """连接设备并运行至成功数达标或 RotationLock 退出。"""

        self._console("[遥操作] 正在连接 SpaceMouse（不会因此发送机器人动作）")
        self.mouse.connect()
        try:
            self._console("[遥操作] 正在连接真机；从腕会执行启动 HOME")
            self.hardware.connect()
            self._gripper_target = self._wait_initial_receipt().action[8]
            first = self._read_and_keepalive()
            self._previous_buttons = dict(first.named_buttons)
            self._recorder = _RecorderThread(
                self.hardware,
                self.project,
                self.output,
                recorder_factory=self._recorder_factory,
                monotonic=self._monotonic,
            )
            self._recorder.start()
            self._console("[遥操作] 已就绪：Menu 开始，Fit 保存，Esc 丢弃，RotationLock 结束")
            while self._successful < self.episodes:
                started = self._monotonic()
                sample = self._read_and_keepalive()
                self._raise_background_failure()
                if self._process_buttons(
                    sample.named_buttons,
                    continuous_inputs_enabled=not bool(sample.stale),
                ):
                    break
                remaining = self._control_period_s - (self._monotonic() - started)
                if remaining > 0.0:
                    self._sleep(remaining)
            return {"successful_episodes": self._successful, "requested_episodes": self.episodes}
        except BaseException as exc:
            self._emergency_reject(f"session failure: {type(exc).__name__}: {exc}")
            with suppress(Exception):
                self.hardware.report_input_failure(exc)
            raise
        finally:
            self._shutdown()

    def _read_and_keepalive(self) -> Any:
        try:
            sample = self.mouse.read(fail_on_stale=False)
            if not bool(sample.connected):
                raise ConnectionError("SpaceMouse 已断开")
            motion = np.asarray(sample.motion, dtype=np.float64).reshape(-1)
            if motion.shape != (6,) or not np.isfinite(motion).all():
                raise ValueError("SpaceMouse 返回非法 motion")
            if sample.stale:
                # 正常静止超过 motion lease 只清零，不等同于设备掉线。
                motion = np.zeros(6, dtype=np.float64)
            if self._recording and any(
                bool(sample.named_buttons.get(name, False)) for name in ("one", "two", "t")
            ):
                raise PermissionError("episode 中禁止 J6 speedJ/HOME")
            modes = self.project["teleop"]["modes"]
            rotation_modifier = str(modes["rotation"]["modifier"])
            wrist_modifier = str(modes["wrist"]["modifier"])
            twist = motion_to_ur5_twist(
                motion,
                sample.named_buttons,
                translation_speed_m_s=float(modes["translation"]["linear_speed_m_s"]),
                rotation_speed_rad_s=float(modes["rotation"]["angular_speed_rad_s"]),
                rotation_button=rotation_modifier,
                suppress_buttons=(wrist_modifier, "rear", "t"),
            )
            wrist_velocity = None if sample.stale else spacemouse_wrist_velocity_deg_s(sample, self.project)
            self._gripper_target = gripper_target_from_buttons(
                self._gripper_target,
                sample,
                self.project["poses"],
            )
            self.hardware.update_command(
                twist,
                wrist_velocity_deg_s=wrist_velocity,
                gripper_target=self._gripper_target,
                issued_monotonic_ns=int(sample.monotonic_ns),
            )
            self.hardware.raise_if_failed()
            return sample
        except BaseException as exc:
            with suppress(Exception):
                self.hardware.report_input_failure(exc)
            raise

    def _process_buttons(
        self,
        buttons: Mapping[str, bool],
        *,
        continuous_inputs_enabled: bool = True,
    ) -> bool:
        names = ("menu", "fit", "esc", "rotation_lock", "t")
        rising = {
            name: bool(buttons.get(name, False)) and not self._previous_buttons.get(name, False)
            for name in names
        }
        self._previous_buttons = dict(buttons)
        one = bool(buttons.get("one", False))
        two = bool(buttons.get("two", False))
        if self._recording and (one or two or rising["t"]):
            self._reject_episode("episode 中按下 J6/T 维护键")
            failure = PermissionError("episode 中禁止 J6 speedJ/HOME")
            self.hardware.report_input_failure(failure)
            raise failure
        # libspnav 没有可靠的 USB-disconnect 事件。motion lease 过期时仍
        # 允许 Menu/Fit 等边沿键工作，但绝不能延续一个可能已经丢失 release
        # 事件的 J6 按住命令；下一次新鲜 motion 到来后才允许继续点动。
        direction = (
            -1
            if continuous_inputs_enabled and one and not two
            else 1
            if continuous_inputs_enabled and two and not one
            else 0
        )
        if self._recording:
            direction = 0
        self.hardware.set_joint6_jog(direction)

        if rising["rotation_lock"]:
            if self._recording:
                self._reject_episode("操作者按 RotationLock 结束采集")
            self._console("[遥操作] 收到 RotationLock，结束采集")
            return True
        if rising["fit"]:
            if self._recording:
                self._save_episode()
            self._full_home("Fit")
            return self._successful >= self.episodes
        if rising["esc"] and self._recording:
            self._reject_episode("操作者按 Esc 丢弃")
            self._gripper_target = float(self.project["poses"]["gripper"]["home_value"])
            self._full_home("Esc")
            return False
        if rising["t"] and not self._recording:
            self._full_home("T")
            return False
        if rising["menu"] and not self._recording:
            self._gripper_target = float(self.project["poses"]["gripper"]["home_value"])
            self._full_home("Menu 起点复位")
            self._start_episode()
        return False

    def _start_episode(self) -> None:
        self.hardware.set_joint6_jog(0)
        self.hardware.set_episode_active(active=True)
        self._recording = True
        try:
            result = self._await_request(self._require_recorder().submit(_RecorderOperation.START))
        except BaseException:
            self._emergency_reject("episode 启动期间发生异常")
            self._recording = False
            with suppress(Exception):
                self.hardware.set_episode_active(active=False)
            raise
        self._console(f"[数采] Episode 已开始：{result}")

    def _save_episode(self) -> None:
        path = self._await_request(self._require_recorder().submit(_RecorderOperation.SAVE))
        self.hardware.set_episode_active(active=False)
        self._recording = False
        self._successful += 1
        self._console(f"[数采] Episode 已原子封存：{path}（成功 {self._successful}/{self.episodes}）")

    def _reject_episode(self, reason: str) -> None:
        path = self._await_request(
            self._require_recorder().submit(_RecorderOperation.REJECT, reason=reason)
        )
        self.hardware.set_episode_active(active=False)
        self._recording = False
        self._console(f"[数采] Episode 已移入 rejected：{path}")

    def _full_home(self, source: str) -> None:
        if self._recording:
            raise RuntimeError("活动 episode 尚未结束，禁止 HOME")
        self.hardware.set_joint6_jog(0)
        self.hardware.request_home()
        self._console(f"[HOME] {source} 请求：从腕机械 HOME + UR task-home")
        deadline = self._monotonic() + self._home_timeout_s
        while self.hardware.maintenance_status().active:
            if self._monotonic() >= deadline:
                raise TimeoutError("等待 full HOME 完成超时")
            self._keepalive_zero_motion()
        self.hardware.request_wrist_master_resume()
        while self.hardware.maintenance_status().active:
            if self._monotonic() >= deadline:
                raise TimeoutError("等待主腕恢复跟随超时")
            self._keepalive_zero_motion()
        self._wait_initial_receipt(deadline=deadline, gripper_target=self._gripper_target)
        self._console("[HOME] 完成，主腕已按当前姿态重新建立跟随零点")

    def _keepalive_zero_motion(self) -> None:
        try:
            sample = self.mouse.read(fail_on_stale=False)
            if not bool(sample.connected):
                raise ConnectionError("零速度 keepalive 期间 SpaceMouse 断开")
            self.hardware.update_command(
                np.zeros(6, dtype=np.float64),
                wrist_velocity_deg_s=None,
                gripper_target=self._gripper_target,
                issued_monotonic_ns=int(sample.monotonic_ns),
            )
            self.hardware.raise_if_failed()
            self._sleep(self._control_period_s)
        except BaseException as exc:
            with suppress(Exception):
                self.hardware.report_input_failure(exc)
            raise

    def _wait_initial_receipt(
        self,
        *,
        deadline: float | None = None,
        gripper_target: float | None = None,
    ) -> ActionReceipt:
        limit = self._monotonic() + 1.0 if deadline is None else deadline
        while True:
            try:
                receipt = self.hardware.latest_action_receipt()
            except RuntimeError:
                receipt = None
            if receipt is not None and (
                gripper_target is None
                or math.isclose(receipt.action[8], gripper_target, abs_tol=1e-6)
            ):
                return receipt
            if self._monotonic() >= limit:
                raise TimeoutError("等待首个可记录 action receipt 超时")
            if gripper_target is None:
                # worker connect 后的零命令租约可维持到首个 receipt 发布。
                self._sleep(min(self._control_period_s, 0.005))
            else:
                self._keepalive_zero_motion()

    def _await_request(self, request: _RecorderRequest, *, timeout_s: float = 15.0) -> Any:
        deadline = self._monotonic() + timeout_s
        while not request.done.is_set():
            if self._monotonic() >= deadline:
                raise TimeoutError(f"等待 recorder {request.operation.value} 超时")
            # 相机 read/HDF5 fsync 再慢，也不能饿死 125 Hz command lease；
            # episode 边界事务期间只发零 speedL，避免未录制的额外位移。
            self._keepalive_zero_motion()
        if request.error is not None:
            raise SessionError(f"recorder {request.operation.value} 失败：{request.error}") from request.error
        return request.result

    def _raise_background_failure(self) -> None:
        self.hardware.raise_if_failed()
        self._require_recorder().raise_if_failed()

    def _emergency_reject(self, reason: str) -> None:
        recorder = self._recorder
        if recorder is None or not self._recording:
            return
        with suppress(Exception):
            request = recorder.submit(_RecorderOperation.REJECT, reason=reason)
            request.done.wait(5.0)
        self._recording = False

    def _shutdown(self) -> None:
        recorder = self._recorder
        if recorder is not None:
            with suppress(Exception):
                recorder.close()
        with suppress(Exception):
            self.hardware.set_episode_active(active=False)
        with suppress(Exception):
            self.hardware.set_joint6_jog(0)
        with suppress(Exception):
            self.hardware.stop()
        with suppress(Exception):
            self.hardware.close()
        with suppress(Exception):
            self.mouse.close()

    def _require_recorder(self) -> _RecorderThread:
        if self._recorder is None:
            raise RuntimeError("recorder 线程尚未启动")
        return self._recorder


__all__ = ["SessionError", "TeleopCollectionSession"]
