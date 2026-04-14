import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformerqec.models.transformer import TransformerQEC
from transformerqec.training.loop import train_step
from transformerqec.training.state import create_train_state


def test_train_step_runs_on_tiny_batch() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=1, ffn_dim=128)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((4, 24), dtype=jnp.float32),
        jnp.full((4,), 0.005, dtype=jnp.float32),
        coords,
    )
    state = create_train_state(
        params=variables["params"],
        apply_fn=model.apply,
        peak_lr=1e-4,
        warmup_steps=0,
        num_steps=4,
    )
    next_state, loss = train_step(
        state,
        jnp.zeros((4, 24), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.full((4,), 0.005, dtype=jnp.float32),
        coords,
        gamma=2.0,
        alpha=0.75,
    )

    before_leaves = jax.tree_util.tree_leaves(variables["params"])
    after_leaves = jax.tree_util.tree_leaves(next_state.params)

    assert int(next_state.step) == 1
    assert float(loss) >= 0.0
    assert any(not np.array_equal(before, after) for before, after in zip(before_leaves, after_leaves))


def test_train_step_runs_with_static_hyperparameters() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=1, ffn_dim=128)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    variables = model.init(
        jax.random.PRNGKey(1),
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.full((2,), 0.005, dtype=jnp.float32),
        coords,
    )
    state = create_train_state(
        params=variables["params"],
        apply_fn=model.apply,
        peak_lr=1e-4,
        warmup_steps=0,
        num_steps=4,
    )

    next_state, loss = train_step(
        state,
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.full((2,), 0.005, dtype=jnp.float32),
        coords,
        gamma=2.0,
        alpha=0.75,
    )

    assert int(next_state.step) == 1
    assert jnp.isfinite(loss)


def test_train_step_rejects_invalid_labels() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=1, ffn_dim=128)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    variables = model.init(
        jax.random.PRNGKey(2),
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.full((2,), 0.005, dtype=jnp.float32),
        coords,
    )
    state = create_train_state(
        params=variables["params"],
        apply_fn=model.apply,
        peak_lr=1e-4,
        warmup_steps=0,
        num_steps=4,
    )

    with pytest.raises(ValueError, match="labels must contain only 0 or 1"):
        train_step(
            state,
            jnp.zeros((2, 24), dtype=jnp.float32),
            jnp.array([0, 2], dtype=jnp.int32),
            jnp.full((2,), 0.005, dtype=jnp.float32),
            coords,
            gamma=2.0,
            alpha=0.75,
        )


def test_train_step_rejects_float_labels() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=1, ffn_dim=128)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    variables = model.init(
        jax.random.PRNGKey(4),
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.full((2,), 0.005, dtype=jnp.float32),
        coords,
    )
    state = create_train_state(
        params=variables["params"],
        apply_fn=model.apply,
        peak_lr=1e-4,
        warmup_steps=0,
        num_steps=4,
    )

    with pytest.raises(ValueError, match="labels must have an integer dtype"):
        train_step(
            state,
            jnp.zeros((2, 24), dtype=jnp.float32),
            jnp.array([0.0, 1.0], dtype=jnp.float32),
            jnp.full((2,), 0.005, dtype=jnp.float32),
            coords,
            gamma=2.0,
            alpha=0.75,
        )


def test_outer_jit_train_step_rejects_invalid_labels() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=1, ffn_dim=128)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    variables = model.init(
        jax.random.PRNGKey(3),
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.full((2,), 0.005, dtype=jnp.float32),
        coords,
    )
    state = create_train_state(
        params=variables["params"],
        apply_fn=model.apply,
        peak_lr=1e-4,
        warmup_steps=0,
        num_steps=4,
    )
    jitted = jax.jit(
        lambda s, syndromes, labels, physical_error_rates, c: train_step(
            s,
            syndromes,
            labels,
            physical_error_rates,
            c,
            gamma=2.0,
            alpha=0.75,
        )
    )

    with pytest.raises(Exception, match="labels must contain only 0 or 1"):
        jitted(
            state,
            jnp.zeros((2, 24), dtype=jnp.float32),
            jnp.array([0, 2], dtype=jnp.int32),
            jnp.full((2,), 0.005, dtype=jnp.float32),
            coords,
        )
