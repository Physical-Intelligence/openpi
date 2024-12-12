import dataclasses

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule:
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    decay_steps: int = int(1e9)
    decay_lr: float = 5e-5

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule:
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


LRScheduleConfig = CosineDecaySchedule | RsqrtDecaySchedule


@dataclasses.dataclass(frozen=True)
class AdamW:
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-4
    clip_gradient_norm: float = 100.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD:
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


OptimizerConfig = AdamW | SGD


def create_optimizer(
    optimizer: OptimizerConfig,
    lr_schedule: LRScheduleConfig,
    weight_decay_mask: at.PyTree | None = None,
    freeze_weights_mask: at.PyTree | None = None,
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    tx = optimizer.create(lr, weight_decay_mask=weight_decay_mask)

    if freeze_weights_mask is not None:
        tx = optax.multi_transform(
            {"online": tx, "frozen": optax.set_to_zero()},
            jax.tree.map(lambda x: "frozen" if x else "online", freeze_weights_mask),
        )

    return tx
