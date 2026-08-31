# ruff: noqa: RUF001, RUF002, RUF003
"""项目内 SpaceMouse 输入、轴映射和按键映射。

导入本模块不会加载 ``spnav`` 或连接 ``spacenavd``。生产后端只在显式
调用 :meth:`SpaceMouse.connect` 时延迟导入；单元测试可注入 fake backend。

默认轴和键码来自已验证旧工程
``src/slai_mi/devices/spacemouse/{buttons,device,spnav}.py``，但都能由外部
mapping/YAML 覆盖。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field
import importlib
import math
import time
from typing import Any, ClassVar

import numpy as np

RAW_AXIS_NAMES = ("x", "y", "z", "rx", "ry", "rz")
DEFAULT_AXIS_MAPPING = ("-z", "+x", "+y", "-rz", "+rx", "+ry")
DEFAULT_BUTTON_CODES = {
    "menu": 0,
    "fit": 1,
    "t": 2,
    "rear": 4,
    "front": 5,
    "roll_cw": 8,
    "one": 12,
    "two": 13,
    "three": 14,
    "four": 15,
    "esc": 22,
    "alt": 23,
    "shift": 24,
    "ctrl": 25,
    "rotation_lock": 26,
}


@dataclass(frozen=True)
class MotionEvent:
    """后端无关的 SpaceMouse 六轴原始事件。"""

    translation: tuple[float, float, float]
    rotation: tuple[float, float, float]


@dataclass(frozen=True)
class ButtonEvent:
    """后端无关的 SpaceMouse 按键事件。"""

    code: int
    pressed: bool


def _float_tuple(values: Any, size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须包含 {size} 个有限数")
    return tuple(float(value) for value in array)


def _parse_axis(token: str) -> tuple[int, float]:
    value = str(token).strip().lower()
    sign = -1.0 if value.startswith("-") else 1.0
    name = value[1:] if value[:1] in {"+", "-"} else value
    if name not in RAW_AXIS_NAMES:
        raise ValueError(f"未知 SpaceMouse 原始轴: {token!r}")
    return RAW_AXIS_NAMES.index(name), sign


@dataclass(frozen=True)
class SpaceMouseConfig:
    """SpaceMouse 归一化、超时和物理映射。"""

    max_raw_value: float = 500.0
    deadzone: tuple[float, ...] = (0.12,) * 6
    stale_timeout_s: float = 0.25
    max_events_per_read: int = 256
    axis_mapping: tuple[str, ...] = DEFAULT_AXIS_MAPPING
    button_codes: Mapping[str, int] = field(default_factory=lambda: DEFAULT_BUTTON_CODES.copy())

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SpaceMouseConfig:
        """从单段 mapping 或 ``load_project_config`` 完整结果构造配置。"""

        source = _flatten_spacemouse_mapping(values)
        if not isinstance(source, Mapping):
            raise TypeError("spacemouse 配置必须是 mapping")
        deadzone_value = source.get("deadzone", (0.12,) * 6)
        if np.asarray(deadzone_value).ndim == 0:
            deadzone_value = (float(deadzone_value),) * 6
        axes = source.get("axis_mapping", DEFAULT_AXIS_MAPPING)
        if isinstance(axes, Mapping):
            axes = axes.get("output", axes.get("normalized_output", DEFAULT_AXIS_MAPPING))
        buttons = source.get("button_codes", DEFAULT_BUTTON_CODES)
        if not isinstance(buttons, Mapping):
            raise TypeError("button_codes 必须是 name: code mapping")
        config = cls(
            max_raw_value=float(source.get("max_raw_value", source.get("max_value", 500.0))),
            deadzone=_float_tuple(deadzone_value, 6, "deadzone"),
            stale_timeout_s=float(
                source.get(
                    "stale_timeout_s",
                    source.get("stale_timeout", float(source.get("stale_timeout_ms", 250.0)) / 1000.0),
                )
            ),
            max_events_per_read=int(source.get("max_events_per_read", 256)),
            axis_mapping=tuple(str(value) for value in axes),
            button_codes={str(name): int(code) for name, code in buttons.items()},
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not math.isfinite(self.max_raw_value) or self.max_raw_value <= 0.0:
            raise ValueError("max_raw_value 必须是正有限数")
        deadzone = np.asarray(self.deadzone, dtype=np.float64)
        if deadzone.shape != (6,) or not np.isfinite(deadzone).all():
            raise ValueError("deadzone 必须包含 6 个有限数")
        if np.any(deadzone < 0.0) or np.any(deadzone >= 1.0):
            raise ValueError("deadzone 必须位于 [0, 1) 内")
        if not math.isfinite(self.stale_timeout_s) or self.stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s 必须是正有限数")
        if not 1 <= self.max_events_per_read <= 4096:
            raise ValueError("max_events_per_read 必须位于 [1, 4096]")
        if len(self.axis_mapping) != 6:
            raise ValueError("axis_mapping 必须有 6 项")
        for token in self.axis_mapping:
            _parse_axis(token)
        codes = [int(code) for code in self.button_codes.values()]
        if any(code < 0 for code in codes) or len(codes) != len(set(codes)):
            raise ValueError("button_codes 必须是互不重复的非负整数")


def _flatten_spacemouse_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """合并 hardware.spacemouse 与 teleop axes/buttons，保持纯解析。"""

    if "hardware" not in values and "teleop" not in values:
        return values.get("spacemouse", values)
    hardware = values.get("hardware", {})
    teleop = values.get("teleop", {})
    if not isinstance(hardware, Mapping) or not isinstance(teleop, Mapping):
        raise TypeError("hardware/teleop 配置必须是 mapping")
    hardware_mouse = hardware.get("spacemouse", {})
    axes = teleop.get("axes", {})
    buttons = teleop.get("buttons", {})
    if not all(isinstance(item, Mapping) for item in (hardware_mouse, axes, buttons)):
        raise TypeError("hardware.spacemouse、teleop.axes/buttons 必须是 mapping")
    missing_buttons = set(DEFAULT_BUTTON_CODES).difference(str(name) for name in buttons)
    if missing_buttons:
        missing = ", ".join(sorted(missing_buttons))
        raise ValueError(f"teleop.buttons 缺少程序 canonical 键名: {missing}")

    output_order = axes.get("output_order", ())
    axis_rules = axes.get("mapping", {})
    if not isinstance(output_order, Sequence) or isinstance(output_order, str):
        raise TypeError("teleop.axes.output_order 必须是列表")
    if not isinstance(axis_rules, Mapping):
        raise TypeError("teleop.axes.mapping 必须是 mapping")
    axis_mapping: list[str] = []
    for output_name in output_order:
        rule = axis_rules.get(output_name)
        if not isinstance(rule, Mapping):
            raise ValueError(f"teleop.axes.mapping 缺少 {output_name}")
        source = str(rule.get("source", ""))
        sign = int(rule.get("sign", 0))
        if sign not in {-1, 1}:
            raise ValueError(f"轴 {output_name} 的 sign 必须为 -1 或 1")
        axis_mapping.append(("-" if sign < 0 else "+") + source)

    button_codes: dict[str, int] = {}
    for canonical_name, item in buttons.items():
        if not isinstance(item, Mapping) or "code" not in item:
            raise ValueError(f"teleop.buttons.{canonical_name} 缺少 code")
        code = int(item["code"])
        # YAML key 必须就是调用方查询的 canonical 名称。这里不根据默认
        # bnum 反推名字，否则用户交换 one/two 的 code 时行为不会真正交换。
        button_codes[str(canonical_name)] = code
    return {
        **hardware_mouse,
        "max_raw_value": axes.get("raw_full_scale", 500.0),
        "deadzone": axes.get("deadzone", 0.12),
        "axis_mapping": axis_mapping,
        "button_codes": button_codes,
    }


def normalize_motion(
    raw_motion: Sequence[float] | np.ndarray,
    config: SpaceMouseConfig,
) -> np.ndarray:
    """把原始 ``[x,y,z,rx,ry,rz]`` 变为配置规定的归一化六轴。"""

    raw = np.asarray(raw_motion, dtype=np.float64).reshape(-1)
    if raw.shape != (6,) or not np.isfinite(raw).all():
        raise ValueError("SpaceMouse 原始轴必须包含 6 个有限数")
    normalized = np.clip(raw / config.max_raw_value, -1.0, 1.0)
    normalized[np.abs(normalized) < np.asarray(config.deadzone)] = 0.0
    result = np.empty(6, dtype=np.float32)
    for output_index, token in enumerate(config.axis_mapping):
        input_index, sign = _parse_axis(token)
        result[output_index] = sign * normalized[input_index]
    return result


@dataclass(frozen=True)
class SpaceMouseSample:
    """一次 SpaceMouse 输入快照。

    ``stale=True`` 时 ``motion`` 已强制清零，因此即使调用者忘记检查也不会
    延续旧速度；安全控制循环仍应在看到 ``healthy=False`` 后停止机器人。
    """

    motion: np.ndarray
    buttons: Mapping[int, bool]
    named_buttons: Mapping[str, bool]
    monotonic_ns: int
    motion_timestamp_ns: int | None
    stale: bool
    connected: bool

    @property
    def healthy(self) -> bool:
        return self.connected and not self.stale and bool(np.isfinite(self.motion).all())

    def pressed(self, name: str) -> bool:
        return bool(self.named_buttons.get(name, False))


class _SpnavBackend:
    """把常见 Python spnav API 适配为 ``open/poll_event/close``。"""

    def __init__(self, module: Any) -> None:
        self._module = module

    def open(self) -> None:
        function = getattr(self._module, "spnav_open", None) or getattr(self._module, "open", None)
        if function is None:
            raise RuntimeError("spnav 模块没有 open/spnav_open")
        result = function()
        if result == -1:
            raise ConnectionError("无法连接 spacenavd")

    def poll_event(self) -> Any | None:
        function = getattr(self._module, "spnav_poll_event", None) or getattr(self._module, "poll_event", None)
        if function is None:
            raise RuntimeError("spnav 模块没有 poll_event/spnav_poll_event")
        return function()

    def close(self) -> None:
        function = getattr(self._module, "spnav_close", None) or getattr(self._module, "close", None)
        if function is not None:
            function()


class _CtypesSpnavBackend:
    """直接使用系统 ``libspnav.so``，协议布局与已验证实现一致。"""

    def __init__(self) -> None:
        # ctypes 和动态库都只在显式 connect 路径中加载，模块导入保持纯净。
        import ctypes

        class Motion(ctypes.Structure):
            _fields_: ClassVar[list[tuple[str, Any]]] = [
                ("type", ctypes.c_int),
                ("x", ctypes.c_int),
                ("y", ctypes.c_int),
                ("z", ctypes.c_int),
                ("rx", ctypes.c_int),
                ("ry", ctypes.c_int),
                ("rz", ctypes.c_int),
                ("period", ctypes.c_uint),
                ("data", ctypes.c_void_p),
            ]

        class Button(ctypes.Structure):
            _fields_: ClassVar[list[tuple[str, Any]]] = [
                ("type", ctypes.c_int),
                ("press", ctypes.c_int),
                ("bnum", ctypes.c_int),
            ]

        class Event(ctypes.Union):
            _fields_: ClassVar[list[tuple[str, Any]]] = [
                ("type", ctypes.c_int),
                ("motion", Motion),
                ("button", Button),
            ]

        self._ctypes = ctypes
        self._event_type = Event
        try:
            self._library = ctypes.CDLL("libspnav.so")
        except OSError as exc:
            raise RuntimeError("缺少系统 libspnav.so；请安装 libspnav-dev/spacenavd") from exc
        self._library.spnav_open.argtypes = []
        self._library.spnav_open.restype = ctypes.c_int
        self._library.spnav_close.argtypes = []
        self._library.spnav_close.restype = None
        self._library.spnav_poll_event.argtypes = [ctypes.POINTER(Event)]
        self._library.spnav_poll_event.restype = ctypes.c_int

    def open(self) -> None:
        if self._library.spnav_open() == -1:
            raise ConnectionError("无法通过 libspnav 连接 spacenavd")

    def poll_event(self) -> MotionEvent | ButtonEvent | None:
        event = self._event_type()
        if self._library.spnav_poll_event(self._ctypes.pointer(event)) == 0:
            return None
        if event.type == 1:
            motion = event.motion
            return MotionEvent(
                (float(motion.x), float(motion.y), float(motion.z)),
                (float(motion.rx), float(motion.ry), float(motion.rz)),
            )
        if event.type == 2:
            return ButtonEvent(code=int(event.button.bnum), pressed=bool(event.button.press))
        raise RuntimeError(f"未知 libspnav 事件类型: {event.type}")

    def close(self) -> None:
        self._library.spnav_close()


def _default_backend() -> _SpnavBackend | _CtypesSpnavBackend:
    """优先使用可选 Python binding，否则使用系统 libspnav。"""

    try:
        module = importlib.import_module("spnav")
    except ModuleNotFoundError:
        return _CtypesSpnavBackend()
    return _SpnavBackend(module)


class SpaceMouse:
    """单所有者、非阻塞 SpaceMouse 读取器。"""

    def __init__(
        self,
        config: SpaceMouseConfig | Mapping[str, Any] | None = None,
        *,
        backend: Any | None = None,
        monotonic_ns: Any = time.monotonic_ns,
    ) -> None:
        if config is None:
            config = SpaceMouseConfig()
        self.config = config if isinstance(config, SpaceMouseConfig) else SpaceMouseConfig.from_mapping(config)
        self.config.validate()
        self._backend = backend
        self._clock_ns = monotonic_ns
        self._connected = False
        self._failure: BaseException | None = None
        self._raw_motion = np.zeros(6, dtype=np.float64)
        self._buttons: dict[int, bool] = {}
        self._last_motion_ns: int | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """连接 spacenavd；不会连接或移动任何机器人。"""

        if self._connected:
            raise RuntimeError("SpaceMouse 已连接")
        if self._backend is None:
            self._backend = _default_backend()
        self._backend.open()
        self._connected = True
        self._failure = None
        self._raw_motion.fill(0.0)
        self._buttons.clear()
        self._last_motion_ns = None

    def read(self, *, fail_on_stale: bool = False) -> SpaceMouseSample:
        """排空当前事件并返回最新快照；过期运动永远清零。

        上层若希望把输入超时直接提升为异常，可设置 ``fail_on_stale=True``。
        """

        if not self._connected:
            raise RuntimeError("SpaceMouse 尚未连接")
        if self._failure is not None:
            raise RuntimeError(f"SpaceMouse 已故障闭锁: {self._failure}") from self._failure
        try:
            for _ in range(self.config.max_events_per_read):
                event = self._backend.poll_event()
                if event is None or (isinstance(event, int) and event == 0):
                    break
                self._consume_event(event)
            now_ns = int(self._clock_ns())
            timeout_ns = int(self.config.stale_timeout_s * 1e9)
            stale = bool(
                self._last_motion_ns is None
                or now_ns < self._last_motion_ns
                or now_ns - self._last_motion_ns > timeout_ns
            )
            motion = np.zeros(6, dtype=np.float32) if stale else normalize_motion(self._raw_motion, self.config)
            named = {name: bool(self._buttons.get(int(code), False)) for name, code in self.config.button_codes.items()}
            sample = SpaceMouseSample(
                motion=motion,
                buttons=self._buttons.copy(),
                named_buttons=named,
                monotonic_ns=now_ns,
                motion_timestamp_ns=self._last_motion_ns,
                stale=stale,
                connected=True,
            )
            if fail_on_stale and stale:
                raise TimeoutError("SpaceMouse 运动输入过期，已清零")
            return sample
        except BaseException as exc:
            self._failure = exc
            self.stop()
            raise

    def read_state(self, *, fail_on_stale: bool = False) -> SpaceMouseSample:
        """``read`` 的语义化别名。"""

        return self.read(fail_on_stale=fail_on_stale)

    def stop(self) -> None:
        """在本地立即清零全部轴和按键。"""

        self._raw_motion.fill(0.0)
        self._buttons.clear()
        self._last_motion_ns = None

    def close(self) -> None:
        """关闭 spnav 连接；可重复调用。"""

        if self._backend is not None and self._connected:
            try:
                self._backend.close()
            finally:
                self._connected = False
                self.stop()

    def healthy(self) -> bool:
        if not self._connected or self._failure is not None or self._last_motion_ns is None:
            return False
        age_ns = int(self._clock_ns()) - self._last_motion_ns
        return 0 <= age_ns <= int(self.config.stale_timeout_s * 1e9)

    def __enter__(self) -> SpaceMouse:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _consume_event(self, event: Any) -> None:
        parsed = _coerce_event(event)
        if isinstance(parsed, MotionEvent):
            raw = np.asarray(parsed.translation + parsed.rotation, dtype=np.float64)
            if raw.shape != (6,) or not np.isfinite(raw).all():
                raise ValueError("SpaceMouse 事件包含 NaN/Inf 或维度错误")
            self._raw_motion[:] = raw
            self._last_motion_ns = int(self._clock_ns())
            return
        if parsed.code < 0:
            raise ValueError("SpaceMouse 按键码必须非负")
        self._buttons[int(parsed.code)] = bool(parsed.pressed)


def _coerce_event(event: Any) -> MotionEvent | ButtonEvent:
    if isinstance(event, MotionEvent | ButtonEvent):
        return event
    if isinstance(event, Mapping):
        if "translation" in event and "rotation" in event:
            return MotionEvent(
                _float_tuple(event["translation"], 3, "translation"),
                _float_tuple(event["rotation"], 3, "rotation"),
            )
        if "bnum" in event or "code" in event:
            return ButtonEvent(
                int(event.get("bnum", event.get("code"))), bool(event.get("press", event.get("pressed")))
            )
    if hasattr(event, "translation") and hasattr(event, "rotation"):
        return MotionEvent(
            _float_tuple(event.translation, 3, "translation"),
            _float_tuple(event.rotation, 3, "rotation"),
        )
    if hasattr(event, "bnum") or hasattr(event, "code"):
        code = getattr(event, "bnum", getattr(event, "code", None))
        pressed = getattr(event, "press", getattr(event, "pressed", False))
        return ButtonEvent(int(code), bool(pressed))
    if all(hasattr(event, name) for name in RAW_AXIS_NAMES):
        return MotionEvent(
            tuple(float(getattr(event, name)) for name in RAW_AXIS_NAMES[:3]),
            tuple(float(getattr(event, name)) for name in RAW_AXIS_NAMES[3:]),
        )
    raise TypeError(f"无法识别 spnav 事件类型: {type(event).__name__}")


def motion_to_ur5_twist(
    motion: Sequence[float] | np.ndarray,
    named_buttons: Mapping[str, bool],
    *,
    translation_speed_m_s: float,
    rotation_speed_rad_s: float,
    rotation_button: str = "shift",
    suppress_buttons: Sequence[str] = ("ctrl", "rear", "t"),
) -> np.ndarray:
    """把归一化帽输入转换为当前实验的 UR5 ``speedL`` 六维命令。

    默认 ``Shift`` 只控制旋转；``Ctrl`` 留给两轴腕、``rear``/``t`` 留给
    回零，所以这些按键按下时 UR 输出为零。平移和旋转分别限制单位范数。
    """

    cap = np.asarray(motion, dtype=np.float64).reshape(-1)
    if cap.shape != (6,) or not np.isfinite(cap).all():
        raise ValueError("归一化 SpaceMouse motion 必须包含 6 个有限数")
    for name, speed in (
        ("translation_speed_m_s", translation_speed_m_s),
        ("rotation_speed_rad_s", rotation_speed_rad_s),
    ):
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError(f"{name} 必须是正有限数")
    twist = np.zeros(6, dtype=np.float64)
    if any(bool(named_buttons.get(name, False)) for name in suppress_buttons):
        return twist
    section = cap[3:].copy() if named_buttons.get(rotation_button, False) else cap[:3].copy()
    norm = float(np.linalg.norm(section))
    if norm > 1.0:
        section /= norm
    if named_buttons.get(rotation_button, False):
        twist[3:] = section * rotation_speed_rad_s
    else:
        twist[:3] = section * translation_speed_m_s
    return twist


__all__ = [
    "DEFAULT_AXIS_MAPPING",
    "DEFAULT_BUTTON_CODES",
    "RAW_AXIS_NAMES",
    "ButtonEvent",
    "MotionEvent",
    "SpaceMouse",
    "SpaceMouseConfig",
    "SpaceMouseSample",
    "motion_to_ur5_twist",
    "normalize_motion",
]
