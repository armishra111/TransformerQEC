"""Pre-Cycle-A baseline model — exact reproduction of commit cc3abc5.

This is the architecture that produced `results/legacy/transformer_qec_d{3,5,7}.pkl`
and `results/evaluation_results.csv`. Defining it as a standalone module here
so scripts/ stays self-contained on the TPU VM (no notebook dependency).

Key properties:
- 2.5D RoPE with seq_len-scaled normalized coordinates (NOT DIPE).
  Coords are normalized to [0, 1] per axis then multiplied by seq_len
  inside `build_rope_2_5d`, so the angular range depends on seq_len.
  This is the original behavior and what the legacy checkpoints expect.
- No locality mask.
- No gradient checkpointing (`nn.remat`). Activations stored in full.
- bfloat16 compute; float32 softmax + classification head.
- Configurable d_model / num_heads / num_layers / ffn_dim (defaults match
  cc3abc5: 128 / 4 / 4 / 1024 = ~1.35M params at d=3).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


def _round_even(n: int) -> int:
    return int(2 * round(n / 2))


def build_rope_2_5d(coords, head_dim: int, seq_len: int,
                    spatial_ratio: int = 3, temporal_ratio: int = 1,
                    base_spatial: float = 10000.0,
                    base_temporal: float = 10000.0):
    """Original (pre-DIPE) RoPE: coords in [0, 1] re-scaled by seq_len."""
    total = spatial_ratio + temporal_ratio

    n_spatial_dims = _round_even(head_dim * spatial_ratio / total)
    n_spatial_dims = max(2, min(n_spatial_dims, head_dim - 2))
    n_temporal_dims = head_dim - n_spatial_dims

    n_spatial_pairs = n_spatial_dims // 2
    n_temporal_pairs = n_temporal_dims // 2

    # Scale [0, 1] coords by seq_len — this is the bit DIPE later removed.
    x_pos = coords[:, 0] * seq_len
    y_pos = coords[:, 1] * seq_len
    t_pos = coords[:, 2] * seq_len

    n_x_pairs = n_spatial_pairs // 2
    n_y_pairs = n_spatial_pairs - n_x_pairs

    freq_x = 1.0 / (base_spatial ** (2.0 * jnp.arange(n_x_pairs) / n_spatial_dims))
    freq_y = 1.0 / (base_spatial ** (2.0 * jnp.arange(n_y_pairs) / n_spatial_dims))

    angles_x = x_pos[:, None] * freq_x[None, :]
    angles_y = y_pos[:, None] * freq_y[None, :]

    min_pairs = min(n_x_pairs, n_y_pairs)
    paired = jnp.stack(
        [angles_x[:, :min_pairs], angles_y[:, :min_pairs]], axis=-1
    )
    interleaved = paired.reshape(angles_x.shape[0], min_pairs * 2)
    parts = [interleaved]
    if n_x_pairs > min_pairs:
        parts.append(angles_x[:, min_pairs:])
    if n_y_pairs > min_pairs:
        parts.append(angles_y[:, min_pairs:])
    angles_spatial = jnp.concatenate(parts, axis=-1) if len(parts) > 1 else interleaved

    freq_t = 1.0 / (base_temporal ** (2.0 * jnp.arange(n_temporal_pairs) / n_temporal_dims))
    angles_temporal = t_pos[:, None] * freq_t[None, :]

    angles = jnp.concatenate([angles_spatial, angles_temporal], axis=-1)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, rope_cos, rope_sin):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * rope_cos - x2 * rope_sin
    out2 = x1 * rope_sin + x2 * rope_cos
    return jnp.concatenate([out1, out2], axis=-1)


class TransformerBlockWithRoPE(nn.Module):
    d_model: int
    num_heads: int
    ffn_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, rope_cos, rope_sin):
        head_dim = self.d_model // self.num_heads

        y = nn.LayerNorm(dtype=self.dtype)(x)

        q = nn.DenseGeneral(features=(self.num_heads, head_dim),
                            axis=-1, dtype=self.dtype, name="query")(y)
        k = nn.DenseGeneral(features=(self.num_heads, head_dim),
                            axis=-1, dtype=self.dtype, name="key")(y)
        v = nn.DenseGeneral(features=(self.num_heads, head_dim),
                            axis=-1, dtype=self.dtype, name="value")(y)

        rc = rope_cos[None, :, None, :]
        rs = rope_sin[None, :, None, :]
        q = apply_rope(q, rc, rs)
        k = apply_rope(k, rc, rs)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        attn_weights = attn_weights.astype(jnp.float32) / scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.astype(self.dtype)
        attn_out = jnp.matmul(attn_weights, v)

        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = nn.DenseGeneral(features=self.d_model,
                                    axis=(-2, -1), dtype=self.dtype, name="out")(attn_out)
        x = x + attn_out

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Dense(self.ffn_dim, dtype=self.dtype)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model, dtype=self.dtype)(y)
        return x + y


class TransformerQEC(nn.Module):
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 1024
    num_classes: int = 2
    pos_encoding: str = "rope"
    rope_spatial_ratio: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, syndrome, p_error, coords):
        B, L = syndrome.shape
        head_dim = self.d_model // self.num_heads

        x = nn.Dense(self.d_model, dtype=self.dtype)(syndrome[..., None])

        cls = self.param(
            "cls_token", nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model)
        )
        cls = cls.astype(self.dtype)
        x = jnp.concatenate(
            [jnp.broadcast_to(cls, (B, 1, self.d_model)), x], axis=1
        )

        p_cond = nn.Dense(self.d_model, dtype=self.dtype)(p_error[:, None])
        p_cond = nn.gelu(p_cond)
        p_cond = nn.Dense(self.d_model, dtype=self.dtype)(p_cond)
        x = x + p_cond[:, None, :]

        rope_cos, rope_sin = build_rope_2_5d(
            coords, head_dim, L,
            spatial_ratio=self.rope_spatial_ratio, temporal_ratio=1,
        )
        rope_cos = rope_cos.astype(self.dtype)
        rope_sin = rope_sin.astype(self.dtype)
        cls_cos = jnp.ones((1, head_dim // 2), dtype=self.dtype)
        cls_sin = jnp.zeros((1, head_dim // 2), dtype=self.dtype)
        rope_cos = jnp.concatenate([cls_cos, rope_cos], axis=0)
        rope_sin = jnp.concatenate([cls_sin, rope_sin], axis=0)

        for _ in range(self.num_layers):
            x = TransformerBlockWithRoPE(
                self.d_model, self.num_heads, self.ffn_dim, dtype=self.dtype
            )(x, rope_cos, rope_sin)

        h = nn.LayerNorm()(x[:, 0].astype(jnp.float32))
        h = nn.Dense(self.d_model)(h)
        h = nn.gelu(h)
        return nn.Dense(self.num_classes)(h)


# Default per-distance configs that match results/legacy/transformer_qec_d{D}.pkl.
LEGACY_CONFIGS: dict[int, dict] = {
    3: {"d_model": 128, "num_heads": 4, "num_layers": 4, "ffn_dim": 1024},
    5: {"d_model": 128, "num_heads": 4, "num_layers": 6, "ffn_dim": 512},
    7: {"d_model": 128, "num_heads": 4, "num_layers": 6, "ffn_dim": 512},
}
