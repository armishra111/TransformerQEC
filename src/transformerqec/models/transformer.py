from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from transformerqec.models.rope import apply_rope, build_rope_2_5d


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ffn_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dtype=self.dtype)(y, y)
        x = x + y

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Dense(self.ffn_dim, dtype=self.dtype)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model, dtype=self.dtype)(y)
        return x + y


class TransformerBlockWithRoPE(nn.Module):
    d_model: int
    num_heads: int
    ffn_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, rope_cos: jnp.ndarray, rope_sin: jnp.ndarray) -> jnp.ndarray:
        head_dim = self.d_model // self.num_heads

        y = nn.LayerNorm(dtype=self.dtype)(x)
        q = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, dtype=self.dtype, name="query")(y)
        k = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, dtype=self.dtype, name="key")(y)
        v = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, dtype=self.dtype, name="value")(y)

        rope_cos = rope_cos[None, :, None, :]
        rope_sin = rope_sin[None, :, None, :]
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

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
        attn_out = nn.DenseGeneral(features=self.d_model, axis=(-2, -1), dtype=self.dtype, name="out")(attn_out)

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
    rope_temporal_ratio: int = 1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, syndrome: jnp.ndarray, p_error: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len = syndrome.shape
        head_dim = self.d_model // self.num_heads

        x = nn.Dense(self.d_model, dtype=self.dtype)(syndrome[..., None])

        if self.pos_encoding != "rope":
            pos = nn.Dense(self.d_model, dtype=self.dtype)(coords)
            pos = nn.gelu(pos)
            pos = nn.Dense(self.d_model, dtype=self.dtype)(pos)
            x = x + pos[None, :, :]

        cls = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.d_model))
        cls = cls.astype(self.dtype)
        x = jnp.concatenate([jnp.broadcast_to(cls, (batch_size, 1, self.d_model)), x], axis=1)

        p_cond = nn.Dense(self.d_model, dtype=self.dtype)(p_error[:, None])
        p_cond = nn.gelu(p_cond)
        p_cond = nn.Dense(self.d_model, dtype=self.dtype)(p_cond)
        x = x + p_cond[:, None, :]

        if self.pos_encoding == "rope":
            rope_cos, rope_sin = build_rope_2_5d(
                coords,
                head_dim,
                seq_len,
                spatial_ratio=self.rope_spatial_ratio,
                temporal_ratio=self.rope_temporal_ratio,
            )
            rope_cos = rope_cos.astype(self.dtype)
            rope_sin = rope_sin.astype(self.dtype)
            cls_cos = jnp.ones((1, head_dim // 2), dtype=self.dtype)
            cls_sin = jnp.zeros((1, head_dim // 2), dtype=self.dtype)
            rope_cos = jnp.concatenate([cls_cos, rope_cos], axis=0)
            rope_sin = jnp.concatenate([cls_sin, rope_sin], axis=0)

            for _ in range(self.num_layers):
                x = TransformerBlockWithRoPE(self.d_model, self.num_heads, self.ffn_dim, dtype=self.dtype)(
                    x, rope_cos, rope_sin
                )
        else:
            for _ in range(self.num_layers):
                x = TransformerBlock(self.d_model, self.num_heads, self.ffn_dim, dtype=self.dtype)(x)

        h = nn.LayerNorm()(x[:, 0].astype(jnp.float32))
        h = nn.Dense(self.d_model)(h)
        h = nn.gelu(h)
        return nn.Dense(self.num_classes)(h)


def build_model_for_distance(config: dict[str, Any]) -> TransformerQEC:
    return TransformerQEC(
        d_model=int(config.get("d_model", 128)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 4)),
        ffn_dim=int(config.get("ffn_dim", 1024)),
        num_classes=int(config.get("num_classes", 2)),
        pos_encoding=str(config.get("pos_encoding", "rope")),
        rope_spatial_ratio=int(config.get("rope_spatial_ratio", config.get("spatial_ratio", 3))),
        rope_temporal_ratio=int(config.get("rope_temporal_ratio", config.get("temporal_ratio", 1))),
    )
