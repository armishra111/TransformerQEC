import math

from flax.training.train_state import TrainState
import optax


def create_optimizer(peak_lr: float, warmup_steps: int, num_steps: int) -> optax.GradientTransformation:
    if not math.isfinite(peak_lr):
        raise ValueError(f"peak_lr must be finite; got {peak_lr}")
    if peak_lr <= 0:
        raise ValueError(f"peak_lr must be > 0; got {peak_lr}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0; got {warmup_steps}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0; got {num_steps}")
    if warmup_steps > num_steps:
        raise ValueError(f"warmup_steps must be <= num_steps; got warmup_steps={warmup_steps}, num_steps={num_steps}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(num_steps, warmup_steps + 1),
        end_value=peak_lr / 10.0,
    )
    return optax.adamw(learning_rate=schedule)


def create_train_state(params, apply_fn, peak_lr: float, warmup_steps: int, num_steps: int) -> TrainState:
    return TrainState.create(apply_fn=apply_fn, params=params, tx=create_optimizer(peak_lr, warmup_steps, num_steps))
