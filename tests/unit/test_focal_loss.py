import jax
import jax.numpy as jnp
import pytest

from transformerqec.training.losses import focal_loss
from transformerqec.training.state import create_optimizer


def test_focal_loss_is_small_for_easy_correct_predictions() -> None:
    logits = jnp.array([[6.0, -6.0], [-6.0, 6.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss = focal_loss(logits, labels, gamma=2.0, alpha=0.75)
    assert float(loss) < 0.01


def test_focal_loss_is_large_for_hard_misclassifications() -> None:
    logits = jnp.array([[-8.0, 8.0], [8.0, -8.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss = focal_loss(logits, labels, gamma=2.0, alpha=0.75)
    grad = jax.grad(lambda x: focal_loss(x, labels, gamma=2.0, alpha=0.75))(logits)

    assert jnp.isfinite(loss)
    assert float(loss) > 5.0
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_focal_loss_matches_balanced_logit_baseline() -> None:
    logits = jnp.array([[0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    loss = focal_loss(logits, labels, gamma=2.0, alpha=0.75)

    assert jnp.isfinite(loss)
    assert float(loss) == pytest.approx(0.0866434, rel=1e-3, abs=1e-5)


def test_focal_loss_rejects_non_binary_logits() -> None:
    logits = jnp.zeros((2, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="logits.shape\\[-1\\] must be 2"):
        focal_loss(logits, labels, gamma=2.0, alpha=0.75)


def test_focal_loss_rejects_label_shape_mismatch() -> None:
    logits = jnp.zeros((2, 2), dtype=jnp.float32)
    labels = jnp.array([[0, 1]], dtype=jnp.int32)

    with pytest.raises(ValueError, match="labels.shape must match logits.shape\\[:-1\\]"):
        focal_loss(logits, labels, gamma=2.0, alpha=0.75)


def test_focal_loss_returns_nan_for_invalid_labels_under_jit() -> None:
    logits = jnp.array([[6.0, -6.0]], dtype=jnp.float32)
    labels = jnp.array([2], dtype=jnp.int32)

    loss = jax.jit(focal_loss)(logits, labels, gamma=2.0, alpha=0.75)

    assert jnp.isnan(loss)


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
