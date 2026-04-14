import jax
import jax.numpy as jnp
import pytest

from transformerqec.training.losses import focal_loss


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


def test_focal_loss_rejects_scalar_logits() -> None:
    logits = jnp.array(0.0, dtype=jnp.float32)
    labels = jnp.array(0, dtype=jnp.int32)

    with pytest.raises(ValueError, match="logits must have rank at least 1"):
        focal_loss(logits, labels, gamma=2.0, alpha=0.75)


def test_focal_loss_rejects_empty_batch() -> None:
    logits = jnp.zeros((0, 2), dtype=jnp.float32)
    labels = jnp.zeros((0,), dtype=jnp.int32)

    with pytest.raises(ValueError, match="labels must not be empty"):
        focal_loss(logits, labels, gamma=2.0, alpha=0.75)


def test_focal_loss_rejects_invalid_labels() -> None:
    logits = jnp.array([[6.0, -6.0]], dtype=jnp.float32)
    labels = jnp.array([2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="labels must contain only 0 or 1"):
        focal_loss(logits, labels, gamma=2.0, alpha=0.75)


def test_outer_jit_focal_loss_rejects_invalid_labels() -> None:
    logits = jnp.array([[6.0, -6.0]], dtype=jnp.float32)
    labels = jnp.array([2], dtype=jnp.int32)
    jitted = jax.jit(lambda x, y: focal_loss(x, y, gamma=2.0, alpha=0.75))

    with pytest.raises(Exception, match="labels must contain only 0 or 1"):
        jitted(logits, labels)


def test_focal_loss_rejects_float_labels() -> None:
    logits = jnp.array([[6.0, -6.0]], dtype=jnp.float32)
    labels = jnp.array([0.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="labels must have an integer dtype"):
        focal_loss(logits, labels, gamma=2.0, alpha=0.75)


def test_vmap_focal_loss_runs_for_valid_labels() -> None:
    logits = jnp.array([[6.0, -6.0], [-6.0, 6.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    losses = jax.vmap(lambda x, y: focal_loss(x, y, gamma=2.0, alpha=0.75))(logits, labels)

    assert losses.shape == (2,)
    assert jnp.all(jnp.isfinite(losses))


def test_jitted_vmap_focal_loss_runs_for_valid_labels() -> None:
    logits = jnp.array([[6.0, -6.0], [-6.0, 6.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    losses = jax.jit(jax.vmap(lambda x, y: focal_loss(x, y, gamma=2.0, alpha=0.75)))(logits, labels)

    assert losses.shape == (2,)
    assert jnp.all(jnp.isfinite(losses))


def test_direct_jit_focal_loss_runs_for_valid_labels() -> None:
    logits = jnp.array([[6.0, -6.0], [-6.0, 6.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    loss = jax.jit(focal_loss)(logits, labels, 2.0, 0.75)

    assert jnp.isfinite(loss)


@pytest.mark.parametrize(
    ("gamma", "message"),
    [(-1.0, "gamma must be >= 0")],
)
def test_focal_loss_rejects_negative_gamma(gamma: float, message: str) -> None:
    logits = jnp.zeros((1, 2), dtype=jnp.float32)
    labels = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=message):
        focal_loss(logits, labels, gamma=gamma, alpha=0.75)


@pytest.mark.parametrize(
    ("gamma", "message"),
    [
        (float("nan"), "gamma must be finite"),
        (float("inf"), "gamma must be finite"),
    ],
)
def test_focal_loss_rejects_non_finite_gamma(gamma: float, message: str) -> None:
    logits = jnp.zeros((1, 2), dtype=jnp.float32)
    labels = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=message):
        focal_loss(logits, labels, gamma=gamma, alpha=0.75)


@pytest.mark.parametrize(
    ("alpha", "message"),
    [
        (-0.1, "alpha must be between 0 and 1 inclusive"),
        (1.1, "alpha must be between 0 and 1 inclusive"),
    ],
)
def test_focal_loss_rejects_out_of_range_alpha(alpha: float, message: str) -> None:
    logits = jnp.zeros((1, 2), dtype=jnp.float32)
    labels = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=message):
        focal_loss(logits, labels, gamma=2.0, alpha=alpha)


@pytest.mark.parametrize(
    ("alpha", "message"),
    [
        (float("nan"), "alpha must be finite"),
        (float("inf"), "alpha must be finite"),
    ],
)
def test_focal_loss_rejects_non_finite_alpha(alpha: float, message: str) -> None:
    logits = jnp.zeros((1, 2), dtype=jnp.float32)
    labels = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=message):
        focal_loss(logits, labels, gamma=2.0, alpha=alpha)


def test_jitted_focal_loss_rejects_invalid_gamma() -> None:
    logits = jnp.array([[6.0, -6.0]], dtype=jnp.float32)
    labels = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(Exception, match="gamma must be finite and within the allowed range"):
        jax.jit(focal_loss)(logits, labels, -1.0, 0.75)
