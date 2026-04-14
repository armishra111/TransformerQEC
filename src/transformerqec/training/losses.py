import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _validate_binary_label_values(labels: Any) -> None:
    if isinstance(labels, jax.core.Tracer):
        return
    labels_array = np.asarray(labels)
    if np.any((labels_array != 0) & (labels_array != 1)):
        raise ValueError("labels must contain only 0 or 1")


def _raise_invalid_labels(_: Any) -> None:
    if not bool(_):
        raise ValueError("labels must contain only 0 or 1")


def _raise_invalid_hyperparameter(flag: Any, *, name: str) -> None:
    if not bool(flag):
        raise ValueError(f"{name} must be finite and within the allowed range")


def _validate_focal_loss_hyperparameters(gamma: float, alpha: float) -> None:
    if isinstance(gamma, jax.core.Tracer) or isinstance(alpha, jax.core.Tracer):
        return
    gamma_value = float(gamma)
    alpha_value = float(alpha)
    if not math.isfinite(gamma_value):
        raise ValueError(f"gamma must be finite; got {gamma}")
    if not math.isfinite(alpha_value):
        raise ValueError(f"alpha must be finite; got {alpha}")
    if gamma_value < 0:
        raise ValueError(f"gamma must be >= 0; got {gamma}")
    if alpha_value < 0 or alpha_value > 1:
        raise ValueError(f"alpha must be between 0 and 1 inclusive; got {alpha}")


def _validate_focal_loss_arguments(logits: jnp.ndarray, labels: jnp.ndarray) -> None:
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError(f"labels must have an integer dtype; got {labels.dtype}")
    if logits.ndim == 0:
        raise ValueError("logits must have rank at least 1; got scalar logits")
    if logits.shape[-1] != 2:
        raise ValueError(f"logits.shape[-1] must be 2; got {logits.shape[-1]}")
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            "labels.shape must match logits.shape[:-1]; "
            f"got labels.shape={labels.shape} and logits.shape[:-1]={logits.shape[:-1]}"
        )
    if labels.size == 0:
        raise ValueError("labels must not be empty")


def _focal_loss_impl(logits: jnp.ndarray, labels: jnp.ndarray, gamma: float, alpha: float) -> jnp.ndarray:
    _validate_focal_loss_arguments(logits, labels)
    valid_labels = jnp.logical_or(labels == 0, labels == 1)
    safe_labels = jnp.where(valid_labels, labels, 0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_p_t = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
    p_t = jnp.exp(log_p_t)
    alpha_t = jnp.where(safe_labels == 1, alpha, 1.0 - alpha)
    loss = jnp.mean(-alpha_t * ((1.0 - p_t) ** gamma) * log_p_t)
    # Keep the runtime checks outside `lax.cond` so they remain transform-safe under `vmap`;
    # `safe_labels` prevents invalid gathers while the callback enforces the public contract.
    jax.debug.callback(_raise_invalid_labels, jnp.all(valid_labels))
    jax.debug.callback(partial(_raise_invalid_hyperparameter, name="gamma"), jnp.isfinite(gamma) & (gamma >= 0))
    jax.debug.callback(
        partial(_raise_invalid_hyperparameter, name="alpha"),
        jnp.isfinite(alpha) & (alpha >= 0) & (alpha <= 1),
    )
    return loss


def focal_loss(logits: jnp.ndarray, labels: jnp.ndarray, gamma: float, alpha: float) -> jnp.ndarray:
    """Compute binary focal loss for 2-class logits and integer labels 0 or 1."""
    _validate_focal_loss_hyperparameters(gamma, alpha)
    _validate_focal_loss_arguments(logits, labels)
    _validate_binary_label_values(labels)
    return _focal_loss_impl(logits, labels, gamma, alpha)
