from __future__ import annotations

import jax.numpy as jnp


def _round_even(value: float) -> int:
    """Round to the nearest even integer."""
    return int(2 * round(value / 2))


def split_rope_dimensions(head_dim: int, spatial_ratio: int, temporal_ratio: int) -> tuple[int, int]:
    """Split a head dimension into spatial and temporal RoPE chunks."""
    total = spatial_ratio + temporal_ratio
    spatial_dim = _round_even(head_dim * spatial_ratio / total)
    spatial_dim = max(2, min(spatial_dim, head_dim - 2))
    temporal_dim = head_dim - spatial_dim
    return spatial_dim, temporal_dim


def _interleave_axis_angles(x_angles: jnp.ndarray, y_angles: jnp.ndarray) -> jnp.ndarray:
    min_pairs = min(x_angles.shape[-1], y_angles.shape[-1])
    paired = jnp.stack([x_angles[:, :min_pairs], y_angles[:, :min_pairs]], axis=-1)
    interleaved = paired.reshape(x_angles.shape[0], min_pairs * 2)
    parts = [interleaved]
    if x_angles.shape[-1] > min_pairs:
        parts.append(x_angles[:, min_pairs:])
    if y_angles.shape[-1] > min_pairs:
        parts.append(y_angles[:, min_pairs:])
    return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else interleaved


def build_rope_2_5d(
    coords: jnp.ndarray,
    head_dim: int,
    seq_len: int,
    spatial_ratio: int = 3,
    temporal_ratio: int = 1,
    base_spatial: float = 10000.0,
    base_temporal: float = 10000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build cosine and sine tables for configurable 2.5D RoPE."""
    spatial_dim, temporal_dim = split_rope_dimensions(head_dim, spatial_ratio, temporal_ratio)
    spatial_pairs = spatial_dim // 2
    temporal_pairs = temporal_dim // 2

    x_pos = coords[:, 0] * seq_len
    y_pos = coords[:, 1] * seq_len
    t_pos = coords[:, 2] * seq_len

    x_pairs = spatial_pairs // 2
    y_pairs = spatial_pairs - x_pairs

    freq_x = 1.0 / (base_spatial ** (2.0 * jnp.arange(x_pairs) / spatial_dim))
    freq_y = 1.0 / (base_spatial ** (2.0 * jnp.arange(y_pairs) / spatial_dim))
    angles_x = x_pos[:, None] * freq_x[None, :]
    angles_y = y_pos[:, None] * freq_y[None, :]
    angles_spatial = _interleave_axis_angles(angles_x, angles_y)

    freq_t = 1.0 / (base_temporal ** (2.0 * jnp.arange(temporal_pairs) / temporal_dim))
    angles_temporal = t_pos[:, None] * freq_t[None, :]

    angles = jnp.concatenate([angles_spatial, angles_temporal], axis=-1)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: jnp.ndarray, rope_cos: jnp.ndarray, rope_sin: jnp.ndarray) -> jnp.ndarray:
    """Apply half-split rotary embeddings to the last dimension of `x`."""
    half = x.shape[-1] // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    out_first = x_first * rope_cos - x_second * rope_sin
    out_second = x_first * rope_sin + x_second * rope_cos
    return jnp.concatenate([out_first, out_second], axis=-1)
