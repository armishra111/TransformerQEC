import jax
import jax.numpy as jnp


def focal_loss(logits: jnp.ndarray, labels: jnp.ndarray, gamma: float, alpha: float) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    one_hot = jax.nn.one_hot(labels, num_classes=2)
    p_t = jnp.sum(probs * one_hot, axis=-1)
    alpha_t = alpha * labels + (1.0 - alpha) * (1 - labels)
    return jnp.mean(-alpha_t * ((1.0 - p_t) ** gamma) * jnp.log(p_t + 1e-8))
