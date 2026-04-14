import jax

from transformerqec.training.losses import focal_loss


@jax.jit
def train_step(state, syndromes, labels, physical_error_rates, coords, gamma: float, alpha: float):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, syndromes, physical_error_rates, coords)
        return focal_loss(logits, labels, gamma=gamma, alpha=alpha)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
