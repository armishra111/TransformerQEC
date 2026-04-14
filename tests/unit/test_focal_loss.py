import jax.numpy as jnp

from transformerqec.training.losses import focal_loss


def test_focal_loss_is_small_for_easy_correct_predictions() -> None:
    logits = jnp.array([[6.0, -6.0], [-6.0, 6.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss = focal_loss(logits, labels, gamma=2.0, alpha=0.75)
    assert float(loss) < 0.01
