import jax

from transformerqec.training.losses import _focal_loss_impl, _validate_binary_label_values


@jax.jit
def _train_step_impl(state, syndromes, labels, physical_error_rates, coords, gamma: float, alpha: float):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, syndromes, physical_error_rates, coords)
        return _focal_loss_impl(logits, labels, gamma=gamma, alpha=alpha)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_step(state, syndromes, labels, physical_error_rates, coords, gamma: float, alpha: float):
    _validate_binary_label_values(labels)
    return _train_step_impl(state, syndromes, labels, physical_error_rates, coords, gamma=gamma, alpha=alpha)
