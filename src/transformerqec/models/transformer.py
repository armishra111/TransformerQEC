from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from transformerqec.models.rope import apply_rope, build_rope_2_5d


SUPPORTED_POSITION_ENCODINGS = frozenset({"rope", "default"})


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")


def _validate_attention_shape(d_model: int, num_heads: int, *, require_rope: bool) -> int:
    _validate_positive("d_model", d_model)
    _validate_positive("num_heads", num_heads)
    if d_model % num_heads != 0:
        raise ValueError(f"d_model must be divisible by num_heads; got {d_model} and {num_heads}")

    head_dim = d_model // num_heads
    if require_rope:
        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE head_dim must be even; "
                f"got d_model={d_model}, num_heads={num_heads}, head_dim={head_dim}"
            )
        if head_dim < 4:
            raise ValueError(
                "RoPE head_dim must be at least 4; "
                f"got d_model={d_model}, num_heads={num_heads}, head_dim={head_dim}"
            )
    return head_dim


def _validate_optional_rope_tables(
    seq_len: int,
    head_dim: int,
    rope_cos: jnp.ndarray | None,
    rope_sin: jnp.ndarray | None,
) -> None:
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rope_cos and rope_sin must either both be provided or both be omitted")
    if rope_cos is None or rope_sin is None:
        return
    expected_shape = (seq_len, head_dim // 2)
    if rope_cos.shape != expected_shape:
        raise ValueError(f"rope_cos must have shape {expected_shape}; got {rope_cos.shape}")
    if rope_sin.shape != expected_shape:
        raise ValueError(f"rope_sin must have shape {expected_shape}; got {rope_sin.shape}")


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
        head_dim = _validate_attention_shape(self.d_model, self.num_heads, require_rope=True)

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

    def _validate_static_config(self) -> int:
        if self.pos_encoding not in SUPPORTED_POSITION_ENCODINGS:
            raise ValueError(
                f"Unsupported pos_encoding {self.pos_encoding!r}; expected one of "
                f"{sorted(SUPPORTED_POSITION_ENCODINGS)!r}"
            )
        _validate_positive("num_layers", self.num_layers)
        _validate_positive("ffn_dim", self.ffn_dim)
        _validate_positive("num_classes", self.num_classes)
        _validate_positive("rope_spatial_ratio", self.rope_spatial_ratio)
        _validate_positive("rope_temporal_ratio", self.rope_temporal_ratio)
        return _validate_attention_shape(
            self.d_model,
            self.num_heads,
            require_rope=self.pos_encoding == "rope",
        )

    @staticmethod
    def _validate_inputs(syndrome: jnp.ndarray, p_error: jnp.ndarray, coords: jnp.ndarray) -> tuple[int, int]:
        if syndrome.ndim != 2:
            raise ValueError(f"syndrome must be rank 2 with shape (batch, length); got shape {syndrome.shape}")
        batch_size, seq_len = syndrome.shape

        if p_error.ndim != 1:
            raise ValueError(f"p_error must be rank 1 with shape (batch,); got shape {p_error.shape}")
        if p_error.shape[0] != batch_size:
            raise ValueError(
                "p_error batch length must match syndrome batch length; "
                f"got {p_error.shape[0]} and {batch_size}"
            )

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (syndrome length, 3); got shape {coords.shape}")
        if coords.shape[0] != seq_len:
            raise ValueError(f"syndrome length must match coords length; got {seq_len} and {coords.shape[0]}")
        return batch_size, seq_len

    @nn.compact
    def __call__(
        self,
        syndrome: jnp.ndarray,
        p_error: jnp.ndarray,
        coords: jnp.ndarray,
        rope_cos: jnp.ndarray | None = None,
        rope_sin: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        head_dim = self._validate_static_config()
        batch_size, seq_len = self._validate_inputs(syndrome, p_error, coords)
        _validate_optional_rope_tables(seq_len, head_dim, rope_cos, rope_sin)

        syndrome = syndrome.astype(self.dtype)
        p_error = p_error.astype(self.dtype)
        coords = coords.astype(self.dtype)

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
            if rope_cos is None or rope_sin is None:
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
