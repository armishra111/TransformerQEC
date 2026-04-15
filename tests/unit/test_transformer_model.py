import jax
import jax.numpy as jnp
import pytest

from transformerqec.models.rope import apply_rope
from transformerqec.models.transformer import TransformerBlockWithRoPE, TransformerQEC


def _init_model(
    model: TransformerQEC,
    syndrome: jnp.ndarray | None = None,
    p_error: jnp.ndarray | None = None,
    coords: jnp.ndarray | None = None,
) -> None:
    model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, 24), dtype=jnp.float32) if syndrome is None else syndrome,
        jnp.array([0.005, 0.01], dtype=jnp.float32) if p_error is None else p_error,
        jnp.zeros((24, 3), dtype=jnp.float32) if coords is None else coords,
    )


def _init_rope_block(block: TransformerBlockWithRoPE) -> None:
    block.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, 25, block.d_model), dtype=jnp.float32),
        jnp.ones((25, 16), dtype=jnp.float32),
        jnp.zeros((25, 16), dtype=jnp.float32),
    )


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


def test_transformer_supports_default_positional_encoding() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=1, ffn_dim=256, pos_encoding="default")
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.array([0.005, 0.01], dtype=jnp.float32),
        coords,
    )
    logits = model.apply(params, jnp.zeros((2, 24), dtype=jnp.float32), jnp.array([0.005, 0.01]), coords)
    assert logits.shape == (2, 2)


def test_transformer_rejects_unsupported_positional_encoding() -> None:
    model = TransformerQEC(pos_encoding="learned")

    with pytest.raises(ValueError, match="Unsupported pos_encoding 'learned'"):
        _init_model(model)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (TransformerQEC(num_heads=0), "num_heads must be positive"),
        (TransformerQEC(d_model=0), "d_model must be positive"),
        (TransformerQEC(d_model=130, num_heads=4), "d_model must be divisible by num_heads"),
        (TransformerQEC(num_layers=0), "num_layers must be positive"),
        (TransformerQEC(num_layers=-1), "num_layers must be positive"),
        (TransformerQEC(ffn_dim=0), "ffn_dim must be positive"),
        (TransformerQEC(num_classes=0), "num_classes must be positive"),
        (TransformerQEC(rope_spatial_ratio=0), "rope_spatial_ratio must be positive"),
        (TransformerQEC(rope_temporal_ratio=0), "rope_temporal_ratio must be positive"),
        (TransformerQEC(d_model=6, num_heads=2), "RoPE head_dim must be even"),
        (TransformerQEC(d_model=4, num_heads=2), "RoPE head_dim must be at least 4"),
    ],
)
def test_transformer_rejects_invalid_model_shapes(model: TransformerQEC, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _init_model(model)


@pytest.mark.parametrize(
    ("syndrome", "p_error", "coords", "message"),
    [
        (
            jnp.zeros((2, 24, 1), dtype=jnp.float32),
            jnp.array([0.005, 0.01], dtype=jnp.float32),
            jnp.zeros((24, 3), dtype=jnp.float32),
            "syndrome must be rank 2",
        ),
        (
            jnp.zeros((2, 24), dtype=jnp.float32),
            jnp.zeros((2, 1), dtype=jnp.float32),
            jnp.zeros((24, 3), dtype=jnp.float32),
            "p_error must be rank 1",
        ),
        (
            jnp.zeros((2, 24), dtype=jnp.float32),
            jnp.array([0.005, 0.01, 0.02], dtype=jnp.float32),
            jnp.zeros((24, 3), dtype=jnp.float32),
            "p_error batch length must match syndrome batch length",
        ),
        (
            jnp.zeros((2, 24), dtype=jnp.float32),
            jnp.array([0.005, 0.01], dtype=jnp.float32),
            jnp.zeros((23, 3), dtype=jnp.float32),
            "syndrome length must match coords length",
        ),
        (
            jnp.zeros((2, 24), dtype=jnp.float32),
            jnp.array([0.005, 0.01], dtype=jnp.float32),
            jnp.zeros((24, 2), dtype=jnp.float32),
            "coords must have shape",
        ),
    ],
)
def test_transformer_rejects_invalid_input_contract(
    syndrome: jnp.ndarray, p_error: jnp.ndarray, coords: jnp.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _init_model(TransformerQEC(), syndrome=syndrome, p_error=p_error, coords=coords)


def test_transformer_block_with_rope_is_exported() -> None:
    from transformerqec.models import TransformerBlockWithRoPE

    assert TransformerBlockWithRoPE.__name__ == "TransformerBlockWithRoPE"


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (TransformerBlockWithRoPE(d_model=128, num_heads=0, ffn_dim=256), "num_heads must be positive"),
        (TransformerBlockWithRoPE(d_model=0, num_heads=4, ffn_dim=256), "d_model must be positive"),
        (
            TransformerBlockWithRoPE(d_model=130, num_heads=4, ffn_dim=256),
            "d_model must be divisible by num_heads",
        ),
        (TransformerBlockWithRoPE(d_model=6, num_heads=2, ffn_dim=256), "RoPE head_dim must be even"),
        (TransformerBlockWithRoPE(d_model=4, num_heads=2, ffn_dim=256), "RoPE head_dim must be at least 4"),
    ],
)
def test_transformer_block_with_rope_rejects_invalid_attention_shapes(
    block: TransformerBlockWithRoPE, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _init_rope_block(block)


def test_apply_rope_supports_notebook_layout_broadcasting() -> None:
    x = jnp.ones((2, 24, 4, 32), dtype=jnp.float32)
    rope_cos = jnp.ones((1, 24, 1, 16), dtype=jnp.float32)
    rope_sin = jnp.zeros((1, 24, 1, 16), dtype=jnp.float32)

    rotated = apply_rope(x, rope_cos, rope_sin)

    assert rotated.shape == x.shape
