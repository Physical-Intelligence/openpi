import enum
import dataclasses

class RTCAttentionSchedule(str, enum.Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"

@dataclasses.dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = False

    # Core RTC settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 1.0
    execution_horizon: int = 10


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(
        self,
        rtc_config: RTCConfig,
        verbose: bool = False,
        visualize_gradients: bool = False,
        viz_output_dir: str = ".",
    ):
        """Initialize RTC processor.

        Args:
            rtc_config (RTCConfig): Configuration holding RTC parameters used by
                the processor, including for example:
                - execution_horizon: number of timesteps used to build prefix weights
                - prefix_attention_schedule: strategy for prefix weights
                  (ZEROS, ONES, LINEAR, EXP)
                - max_guidance_weight: upper bound applied to the guidance scale
            verbose (bool): Enable verbose debug logging.
            visualize_gradients (bool): Enable gradient visualization using torchviz.
            viz_output_dir (str): Directory to save gradient visualizations.
        """
        self.rtc_config = rtc_config

    def get_prefix_weights(start: int, end: int, total: int, schedule: RTCAttentionSchedule) -> jax.Array:
        """With start=2, end=6, total=10, the output will be:
        1  1  4/5 3/5 2/5 1/5 0  0  0  0
            ^              ^
            start           end
        `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
        paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
        entire prefix is attended to.

        `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
        if `end` is 0, then the entire prefix will always be ignored.
        """
        start = jnp.minimum(start, end)
        if schedule == "ones":
            w = jnp.ones(total)
        elif schedule == "zeros":
            w = (jnp.arange(total) < start).astype(jnp.float32)
        elif schedule == "linear" or schedule == "exp":
            w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
            if schedule == "exp":
                w = w * jnp.expm1(w) / (jnp.e - 1)
        else:
            raise ValueError(f"Invalid schedule: {schedule}")
        return jnp.where(jnp.arange(total) >= end, 0, w)