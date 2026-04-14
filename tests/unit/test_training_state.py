import pytest

from transformerqec.training.state import create_optimizer


@pytest.mark.parametrize(
    ("peak_lr", "warmup_steps", "num_steps", "message"),
    [
        (0.0, 1, 4, "peak_lr must be > 0"),
        (-1e-4, 1, 4, "peak_lr must be > 0"),
        (1e-4, -1, 4, "warmup_steps must be >= 0"),
        (1e-4, 1, 0, "num_steps must be > 0"),
        (1e-4, 5, 4, "warmup_steps must be <= num_steps"),
    ],
)
def test_create_optimizer_rejects_invalid_arguments(
    peak_lr: float, warmup_steps: int, num_steps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_optimizer(peak_lr=peak_lr, warmup_steps=warmup_steps, num_steps=num_steps)


@pytest.mark.parametrize(
    ("peak_lr", "message"),
    [
        (float("nan"), "peak_lr must be finite"),
        (float("inf"), "peak_lr must be finite"),
    ],
)
def test_create_optimizer_rejects_non_finite_peak_lr(peak_lr: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_optimizer(peak_lr=peak_lr, warmup_steps=1, num_steps=4)
