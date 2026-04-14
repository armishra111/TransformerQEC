import jax
import jax.numpy as jnp

from transformerqec.models.rope import apply_rope
from transformerqec.models.transformer import TransformerQEC


def test_transformer_forward_shape() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=4, ffn_dim=1024)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.array([0.005, 0.01], dtype=jnp.float32),
        coords,
    )
    logits = model.apply(params, jnp.zeros((2, 24), dtype=jnp.float32), jnp.array([0.005, 0.01]), coords)
    assert logits.shape == (2, 2)


def test_apply_rope_supports_notebook_layout_broadcasting() -> None:
    x = jnp.ones((2, 24, 4, 32), dtype=jnp.float32)
    rope_cos = jnp.ones((1, 24, 1, 16), dtype=jnp.float32)
    rope_sin = jnp.zeros((1, 24, 1, 16), dtype=jnp.float32)

    rotated = apply_rope(x, rope_cos, rope_sin)

    assert rotated.shape == x.shape
