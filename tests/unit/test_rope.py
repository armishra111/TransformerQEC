import jax.numpy as jnp

from transformerqec.models.rope import apply_rope, build_rope_2_5d, split_rope_dimensions


def test_rope_dimension_split_preserves_head_dim() -> None:
    spatial_dim, temporal_dim = split_rope_dimensions(head_dim=128, spatial_ratio=3, temporal_ratio=1)
    assert spatial_dim == 96
    assert temporal_dim == 32


def test_rope_tables_match_sequence_length() -> None:
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    rope_cos, rope_sin = build_rope_2_5d(coords=coords, head_dim=32, seq_len=24, spatial_ratio=3, temporal_ratio=1)
    assert rope_cos.shape == (24, 16)
    assert rope_sin.shape == (24, 16)


def test_apply_rope_preserves_tensor_shape() -> None:
    x = jnp.ones((2, 4, 24, 32), dtype=jnp.float32)
    rope_cos = jnp.ones((24, 16), dtype=jnp.float32)
    rope_sin = jnp.zeros((24, 16), dtype=jnp.float32)
    rotated = apply_rope(x, rope_cos, rope_sin)
    assert rotated.shape == x.shape


def test_spatial_angles_interleave_x_and_y_bands() -> None:
    coords = jnp.array([[1.0, 2.0, 0.0]], dtype=jnp.float32)
    rope_cos, rope_sin = build_rope_2_5d(
        coords=coords,
        head_dim=16,
        seq_len=1,
        spatial_ratio=3,
        temporal_ratio=1,
    )

    x_pos = coords[:, 0] * 1
    y_pos = coords[:, 1] * 1
    n_spatial_dims = 12
    n_x_pairs = 3
    n_y_pairs = 3
    freq_x = 1.0 / (10000.0 ** (2.0 * jnp.arange(n_x_pairs) / n_spatial_dims))
    freq_y = 1.0 / (10000.0 ** (2.0 * jnp.arange(n_y_pairs) / n_spatial_dims))

    expected = jnp.array(
        [
            jnp.sin(x_pos[0] * freq_x[0]),
            jnp.sin(y_pos[0] * freq_y[0]),
            jnp.sin(x_pos[0] * freq_x[1]),
            jnp.sin(y_pos[0] * freq_y[1]),
            jnp.sin(x_pos[0] * freq_x[2]),
            jnp.sin(y_pos[0] * freq_y[2]),
        ],
        dtype=jnp.float32,
    )

    assert jnp.allclose(rope_sin[0, :6], expected)
