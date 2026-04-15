import jax.numpy as jnp
import numpy as np
import pytest

from transformerqec.models.rope import apply_rope, build_rope_2_5d, get_rope_tables, split_rope_dimensions


def test_rope_dimension_split_preserves_head_dim() -> None:
    spatial_dim, temporal_dim = split_rope_dimensions(head_dim=128, spatial_ratio=3, temporal_ratio=1)
    assert spatial_dim == 96
    assert temporal_dim == 32


def test_rope_dimension_split_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="head_dim must be even"):
        split_rope_dimensions(head_dim=127, spatial_ratio=3, temporal_ratio=1)

    with pytest.raises(ValueError, match="head_dim must be at least 4"):
        split_rope_dimensions(head_dim=2, spatial_ratio=3, temporal_ratio=1)

    with pytest.raises(ValueError, match="spatial_ratio must be positive"):
        split_rope_dimensions(head_dim=128, spatial_ratio=0, temporal_ratio=1)

    with pytest.raises(ValueError, match="temporal_ratio must be positive"):
        split_rope_dimensions(head_dim=128, spatial_ratio=3, temporal_ratio=0)


def test_rope_tables_match_sequence_length() -> None:
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    rope_cos, rope_sin = build_rope_2_5d(coords=coords, head_dim=32, seq_len=24, spatial_ratio=3, temporal_ratio=1)
    assert rope_cos.shape == (24, 16)
    assert rope_sin.shape == (24, 16)


def test_build_rope_rejects_invalid_bases_and_coords_shape() -> None:
    coords = jnp.zeros((24, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="base_spatial must be positive"):
        build_rope_2_5d(coords=coords, head_dim=32, seq_len=24, base_spatial=0.0)

    with pytest.raises(ValueError, match="base_temporal must be positive"):
        build_rope_2_5d(coords=coords, head_dim=32, seq_len=24, base_temporal=0.0)

    with pytest.raises(ValueError, match="coords must have shape \\(L, 3\\)"):
        build_rope_2_5d(coords=jnp.zeros((24, 2), dtype=jnp.float32), head_dim=32, seq_len=24)


def test_apply_rope_preserves_tensor_shape() -> None:
    x = jnp.ones((2, 4, 24, 32), dtype=jnp.float32)
    rope_cos = jnp.ones((24, 16), dtype=jnp.float32)
    rope_sin = jnp.zeros((24, 16), dtype=jnp.float32)
    rotated = apply_rope(x, rope_cos, rope_sin)
    assert rotated.shape == x.shape


def test_apply_rope_rotates_values_with_nonzero_sine() -> None:
    x = jnp.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=jnp.float32)
    rope_cos = jnp.array([[0.6, 0.8]], dtype=jnp.float32)
    rope_sin = jnp.array([[0.8, 0.6]], dtype=jnp.float32)

    rotated = apply_rope(x, rope_cos, rope_sin)

    expected = jnp.array([[[[-1.8, -0.8, 2.6, 4.4]]]], dtype=jnp.float32)
    assert jnp.allclose(rotated, expected)


def test_spatial_angles_interleave_x_and_y_bands() -> None:
    coords = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    rope_cos, rope_sin = build_rope_2_5d(
        coords=coords,
        head_dim=16,
        seq_len=1,
        spatial_ratio=3,
        temporal_ratio=1,
    )

    expected_angles = jnp.array(
        [
            1.0,
            2.0,
            1.0 / (10000.0 ** (2.0 / 12.0)),
            2.0 / (10000.0 ** (2.0 / 12.0)),
            1.0 / (10000.0 ** (4.0 / 12.0)),
            2.0 / (10000.0 ** (4.0 / 12.0)),
            3.0,
            3.0 / (10000.0 ** (2.0 / 4.0)),
        ],
        dtype=jnp.float32,
    )

    assert jnp.allclose(rope_sin[0], jnp.sin(expected_angles))
    assert jnp.allclose(rope_cos[0], jnp.cos(expected_angles))


def test_get_rope_tables_reuses_cached_tables_for_equivalent_inputs() -> None:
    coords = np.zeros((24, 3), dtype=np.float32)

    first_cos, first_sin = get_rope_tables(coords, head_dim=32, seq_len=24, spatial_ratio=3, temporal_ratio=1)
    second_cos, second_sin = get_rope_tables(
        jnp.zeros((24, 3), dtype=jnp.float32),
        head_dim=32,
        seq_len=24,
        spatial_ratio=3,
        temporal_ratio=1,
    )

    assert first_cos is second_cos
    assert first_sin is second_sin
