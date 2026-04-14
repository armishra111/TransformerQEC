import jax
import jax.numpy as jnp


def focal_loss(logits: jnp.ndarray, labels: jnp.ndarray, gamma: float, alpha: float) -> jnp.ndarray:
    """Compute binary focal loss for logits with two classes and integer labels."""
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0; got {gamma}")
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be between 0 and 1 inclusive; got {alpha}")
    if logits.shape[-1] != 2:
        raise ValueError(f"logits.shape[-1] must be 2; got {logits.shape[-1]}")
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            "labels.shape must match logits.shape[:-1]; "
            f"got labels.shape={labels.shape} and logits.shape[:-1]={logits.shape[:-1]}"
        )

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    valid_labels = jnp.logical_or(labels == 0, labels == 1)
    safe_labels = jnp.where(valid_labels, labels, 0)
    log_p_t = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
    p_t = jnp.exp(log_p_t)
    alpha_t = jnp.where(labels == 1, alpha, 1.0 - alpha)
    loss = -alpha_t * ((1.0 - p_t) ** gamma) * log_p_t
    loss = jnp.where(valid_labels, loss, jnp.nan)
    return jnp.mean(loss)
