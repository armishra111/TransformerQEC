import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# 2.5D Rotary Position Embedding (RoPE) for QEC space-time coordinates
# ---------------------------------------------------------------------------
# Encodes [x,y] jointly via 2D RoPE and [t] via 1D RoPE into separate
# subspaces of each attention head.  After the standard Q @ K^T dot
# product, the score decomposes as:
#
#   Attention(i,j) ~ Q_{xy,i} . K_{xy,j}  +  Q_{t,i} . K_{t,j}
#
# because the spatial and temporal dimension chunks are orthogonal.
# Each chunk uses RoPE's rotation, so the score depends only on the
# *relative* displacement (dx, dy, dt) — translational invariance.
# ---------------------------------------------------------------------------

def _round_even(n):
    """Round to nearest even integer (RoPE needs dimension pairs)."""
    return int(2 * round(n / 2))


def build_rope_2_5d(coords, head_dim,
                    spatial_ratio=3, temporal_ratio=1,
                    base_spatial=100.0, base_temporal=20.0):
    """Build cos/sin tables for 2.5D RoPE from raw integer detector coords.

    DIPE: positions are the raw (origin-shifted) integer detector coordinates
    from STIM. Identical (x, y, t) at any code distance produces the identical
    RoPE rotation, so the same physical lattice cell receives the same angle
    regardless of d. This restores translational invariance across distances.

    Base frequencies are tuned for the small positional ranges used by surface
    code detectors (max spatial ~2(d-1), max temporal ~d), where the standard
    NLP base of 10000 leaves the lower-frequency bands carrying no signal.

    Args:
        coords:         (L, 3) raw integer detector coords [x, y, t], origin-shifted to >= 0.
        head_dim:       int, per-head dimension (must be even).
        spatial_ratio:  int, relative weight for spatial dims (default 3).
        temporal_ratio: int, relative weight for temporal dims (default 1).
        base_spatial:   float, RoPE frequency base for [x,y] (default 100).
        base_temporal:  float, RoPE frequency base for [t] (default 20).

    Returns:
        (rope_cos, rope_sin) each of shape (L, head_dim//2).
    """
    total = spatial_ratio + temporal_ratio

    # --- dynamic split, forced to even dim counts ---
    n_spatial_dims = _round_even(head_dim * spatial_ratio / total)
    n_spatial_dims = max(2, min(n_spatial_dims, head_dim - 2))  # at least 1 pair each
    n_temporal_dims = head_dim - n_spatial_dims

    n_spatial_pairs = n_spatial_dims // 2
    n_temporal_pairs = n_temporal_dims // 2

    # Raw integer detector coordinates — no per-distance rescaling (DIPE).
    x_pos = coords[:, 0]
    y_pos = coords[:, 1]
    t_pos = coords[:, 2]

    # =================================================================
    # Spatial chunk — 2D RoPE interleaving x and y
    # =================================================================
    # Allocate half the spatial pairs to x, half to y.  For each axis
    # we compute independent inverse-frequency bands, then interleave
    # the (x, y) angles so nearby spatial pairs encode the two axes at
    # the same frequency scale.  This joint encoding couples (x,y)
    # within the spatial subspace.
    n_x_pairs = n_spatial_pairs // 2
    n_y_pairs = n_spatial_pairs - n_x_pairs  # may be +1 if odd

    # Inverse-frequency bands: theta_i = 1 / base^(2i / n_spatial_dims)
    # Each axis gets its own progression so the frequency spectrum
    # covers the full range independently.
    freq_x = 1.0 / (base_spatial ** (2.0 * jnp.arange(n_x_pairs) / n_spatial_dims))
    freq_y = 1.0 / (base_spatial ** (2.0 * jnp.arange(n_y_pairs) / n_spatial_dims))

    # Outer products: position * frequency -> rotation angles
    angles_x = x_pos[:, None] * freq_x[None, :]   # (L, n_x_pairs)
    angles_y = y_pos[:, None] * freq_y[None, :]   # (L, n_y_pairs)

    # Vectorized interleave: stack + reshape replaces O(n) slice ops with 1 XLA op.
    min_pairs = min(n_x_pairs, n_y_pairs)
    paired = jnp.stack([angles_x[:, :min_pairs],
                        angles_y[:, :min_pairs]], axis=-1)        # (L, min_pairs, 2)
    interleaved = paired.reshape(angles_x.shape[0], min_pairs * 2)  # (L, min_pairs*2)
    # Append any remaining from the longer axis (at most 1 extra pair)
    parts = [interleaved]
    if n_x_pairs > min_pairs:
        parts.append(angles_x[:, min_pairs:])
    if n_y_pairs > min_pairs:
        parts.append(angles_y[:, min_pairs:])
    angles_spatial = jnp.concatenate(parts, axis=-1) if len(parts) > 1 else interleaved

    # =================================================================
    # Temporal chunk — standard 1D RoPE on t
    # =================================================================
    freq_t = 1.0 / (base_temporal ** (2.0 * jnp.arange(n_temporal_pairs) / n_temporal_dims))
    angles_temporal = t_pos[:, None] * freq_t[None, :]  # (L, n_temporal_pairs)

    # =================================================================
    # Concatenate spatial + temporal angles -> full (L, head_dim//2) table
    # =================================================================
    angles = jnp.concatenate([angles_spatial, angles_temporal], axis=-1)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, rope_cos, rope_sin):
    """Apply rotary embedding to the last dimension of x.

    Uses the half-split layout (LLaMA-style):
      x = [x_first_half | x_second_half]
    Each corresponding element pair (x_first[i], x_second[i]) is rotated
    by the angle at position i:
      out_first[i]  =  x_first[i] * cos[i]  -  x_second[i] * sin[i]
      out_second[i] =  x_first[i] * sin[i]  +  x_second[i] * cos[i]

    This is mathematically equivalent to the complex-number formulation
    but avoids complex dtypes for XLA compatibility.

    Args:
        x:        (..., head_dim)  — Q or K tensor.
        rope_cos: (..., head_dim//2) — cosine table (broadcastable).
        rope_sin: (..., head_dim//2) — sine table (broadcastable).

    Returns:
        Rotated tensor, same shape as x.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # first half of dims
    x2 = x[..., half:]   # second half of dims
    out1 = x1 * rope_cos - x2 * rope_sin
    out2 = x1 * rope_sin + x2 * rope_cos
    return jnp.concatenate([out1, out2], axis=-1)


# ---------------------------------------------------------------------------
# Transformer block with RoPE
# ---------------------------------------------------------------------------

class TransformerBlockWithRoPE(nn.Module):
    """Pre-norm transformer block with manual attention for RoPE injection.

    RoPE must be applied to Q and K *after* linear projection but
    *before* the dot product.  Flax's MultiHeadDotProductAttention does
    not expose this hook point, so we project Q/K/V manually via
    DenseGeneral and compute scaled dot-product attention inline.
    """
    d_model: int
    num_heads: int
    ffn_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, rope_cos, rope_sin):
        """
        Args:
            x:         (B, L, d_model)
            rope_cos:  (L, head_dim//2)  — precomputed cos table
            rope_sin:  (L, head_dim//2)  — precomputed sin table
        """
        head_dim = self.d_model // self.num_heads

        # --- Pre-norm self-attention with RoPE ---
        y = nn.LayerNorm(dtype=self.dtype)(x)

        # Project to multi-head Q, K, V: (B, L, num_heads, head_dim)
        q = nn.DenseGeneral(features=(self.num_heads, head_dim),
                            axis=-1, dtype=self.dtype, name='query')(y)
        k = nn.DenseGeneral(features=(self.num_heads, head_dim),
                            axis=-1, dtype=self.dtype, name='key')(y)
        v = nn.DenseGeneral(features=(self.num_heads, head_dim),
                            axis=-1, dtype=self.dtype, name='value')(y)

        # Apply RoPE to Q and K.
        # rope tables: (L, half) -> broadcast to (1, L, 1, half) over B, H
        rc = rope_cos[None, :, None, :]   # (1, L, 1, half)
        rs = rope_sin[None, :, None, :]
        q = apply_rope(q, rc, rs)
        k = apply_rope(k, rc, rs)

        # Scaled dot-product attention
        # Transpose to (B, H, L, D) for batched matmul
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Precision discipline: QK matmul in compute dtype (bf16 on TPU MXU),
        # upcast to f32 for softmax stability, then back to compute dtype
        # for V matmul — keeps both MXU matmuls in bf16.
        scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1))   # bf16 @ bf16 on MXU
        attn_weights = attn_weights.astype(jnp.float32) / scale   # f32 for softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.astype(self.dtype)             # back to compute dtype
        attn_out = jnp.matmul(attn_weights, v)    # bf16 @ bf16 on MXU

        # Back to (B, L, H, D) then project to d_model
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = nn.DenseGeneral(features=self.d_model,
                                   axis=(-2, -1), dtype=self.dtype, name='out')(attn_out)

        x = x + attn_out

        # --- Pre-norm feed-forward (identical to standard block) ---
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Dense(self.ffn_dim, dtype=self.dtype)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model, dtype=self.dtype)(y)
        return x + y


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TransformerQEC(nn.Module):
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 1024
    num_classes: int = 2
    rope_spatial_ratio: int = 3  # spatial:temporal dim ratio for RoPE
    dtype: jnp.dtype = jnp.float32
    # DIPE: raw integer coords drive RoPE; these fields are unused at runtime
    # but retained so legacy checkpoint config dicts load without error.
    code_distance: int = 3
    measurement_rounds: int = 3

    @nn.compact
    def __call__(self, syndrome, p_error, coords):
        """
        Args:
            syndrome: (B, L) binary detection events
            p_error:  (B,) physical error rates
            coords:   (L, 3) raw integer detector coordinates (x, y, round),
                      origin-shifted so each axis starts at 0.
        """
        B, L = syndrome.shape
        head_dim = self.d_model // self.num_heads

        # Embed each binary detection event
        x = nn.Dense(self.d_model, dtype=self.dtype)(syndrome[..., None])  # (B, L, d_model)

        # Prepend learnable CLS token
        # Cast CLS from f32 param to compute dtype to prevent
        # silent upcast when concatenated with bf16 sequence
        cls = self.param('cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.d_model))
        cls = cls.astype(self.dtype)
        x = jnp.concatenate(
            [jnp.broadcast_to(cls, (B, 1, self.d_model)), x], axis=1)

        # Condition on physical error rate
        p_cond = nn.Dense(self.d_model, dtype=self.dtype)(p_error[:, None])
        p_cond = nn.gelu(p_cond)
        p_cond = nn.Dense(self.d_model, dtype=self.dtype)(p_cond)
        x = x + p_cond[:, None, :]  # broadcast to all tokens

        # --- Build 2.5D RoPE tables from raw integer detector coordinates ---
        rope_cos, rope_sin = build_rope_2_5d(
            coords, head_dim,
            spatial_ratio=self.rope_spatial_ratio, temporal_ratio=1)
        # Cast from f32 (trig precision) to compute dtype for MXU
        rope_cos = rope_cos.astype(self.dtype)
        rope_sin = rope_sin.astype(self.dtype)
        # Prepend identity rotation for CLS token (no physical position):
        # cos=1, sin=0 means Q/K pass through unrotated — CLS attends
        # everywhere without positional bias.
        cls_cos = jnp.ones((1, head_dim // 2), dtype=self.dtype)
        cls_sin = jnp.zeros((1, head_dim // 2), dtype=self.dtype)
        rope_cos = jnp.concatenate([cls_cos, rope_cos], axis=0)  # (L+1, half)
        rope_sin = jnp.concatenate([cls_sin, rope_sin], axis=0)

        # --- Transformer encoder stack ---
        # nn.remat: block activations recomputed in backward pass instead of
        # stored. ~O(1/sqrt(num_layers)) activation memory at ~30% extra
        # compute. Forward output is bit-identical — checkpoints compatible
        # with and without remat.
        block_cls = nn.remat(TransformerBlockWithRoPE)
        for _ in range(self.num_layers):
            x = block_cls(
                self.d_model, self.num_heads, self.ffn_dim,
                dtype=self.dtype)(x, rope_cos, rope_sin)

        # Classification head: float32 for numerical stability
        h = nn.LayerNorm()(x[:, 0].astype(jnp.float32))
        h = nn.Dense(self.d_model)(h)
        h = nn.gelu(h)
        return nn.Dense(self.num_classes)(h)
