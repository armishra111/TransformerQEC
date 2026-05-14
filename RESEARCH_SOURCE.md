# RESEARCH_SOURCE

## 1. PROJECT ESSENCE
Train 1.3M param Transformer encoder to decode rotated surface codes under phenomenological noise. Input: syndrome bit-string $\mathbf{s} \in \{0, 1\}^{N_d}$ and physical error rate $p$. Output: logical $\hat{Z}_L$ error prediction $\hat{y} = f_\theta(\mathbf{s}, p) \in \{0, 1\}$. Outperforms MWPM below threshold for $d=3, 5$. Generalizes to $d=7$ zero-shot but hits capacity limits. Goal: Scale capability via physics-informed inductive biases (not param bloat).

## 2. CURRENT MATHEMATICAL STATE
**Focal Loss (Counter class imbalance at low $p$):**
$$\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t \,(1 - p_t)^\gamma \,\log(p_t) \quad (\gamma=2.0, \alpha=0.75)$$

**(2+1)D Anisotropic RoPE Attention (Orthogonal spatial/temporal bases):**
$$A(i, j) \propto \exp\left(\frac{\mathbf{q}^{(xy)}_i \cdot \mathbf{k}^{(xy)}_j + \mathbf{q}^{(t)}_i \cdot \mathbf{k}^{(t)}_j}{\sqrt{d_{\text{head}}}}\right)$$

**Surface Code Power-Law Scaling:**
$$P_L = C \cdot (p/p_{\text{th}})^{(d+1)/2}$$

**Calibrated Physics Penalty (Proposed Attention Mask):**
$$AttentionScore - \beta \cdot [-\log(p)] \cdot GraphDistance$$

## 3. TECHNICAL BOTTLENECKS
- **Parameter Capacity Bound:** 1.3M params insufficient for combinatorially rich $d=7$ syndrome volume (336 detectors).
- **Inference Latency:** $O(L^2)$ Transformer attention vs highly optimized $O(L^3)$ MWPM C++ implementations.
- **RoPE Angular Drift:** Sequence-length dependent coordinate scaling breaks relative rotational invariants across varying code distances.
- **Topological vs Coherent Noise:** Hard to disentangle local stochastic topological errors (true qubit noise) from low-frequency global coherent shifts (environmental/hardware leakage).

## 4. CODE SNIPPETS (CRITICAL KERNELS)

**(2+1)D RoPE Frequency Allocation:**
```python
def build_rope_2_5d(coords, head_dim, code_distance, measurement_rounds, spatial_ratio=3, temporal_ratio=1, base_spatial=10000.0, base_temporal=10000.0):
    # Scale coords from [0,1] to integer-like range
    x_pos = coords[:, 0] * code_distance
    y_pos = coords[:, 1] * code_distance
    t_pos = coords[:, 2] * measurement_rounds
    
    # Independent inverse-frequency bands for spatial (x,y) and temporal (t) axes
    freq_x = 1.0 / (base_spatial ** (2.0 * jnp.arange(n_x_pairs) / n_spatial_dims))
    freq_y = 1.0 / (base_spatial ** (2.0 * jnp.arange(n_y_pairs) / n_spatial_dims))
    freq_t = 1.0 / (base_temporal ** (2.0 * jnp.arange(n_temporal_pairs) / n_temporal_dims))
    
    # Interleave x and y angles for joint spatial encoding
    # ... (see model.py for vectorized interleaving logic)
    return jnp.cos(angles), jnp.sin(angles)
```

**RoPE Attention Injection (Pre-Softmax):**
```python
# Project to multi-head Q, K, V
q = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, dtype=self.dtype)(y)
k = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, dtype=self.dtype)(y)

# Apply RoPE to Q and K separately
q = apply_rope(q, rc, rs)
k = apply_rope(k, rc, rs)

# Scaled dot-product attention (bf16 -> f32 -> bf16 precision discipline)
scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
attn_weights = attn_weights.astype(jnp.float32) / scale

if mask is not None:
    attn_weights = jnp.where(mask, attn_weights, -1e9)

attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(self.dtype)
attn_out = jnp.matmul(attn_weights, v)
```

## 5. RESEARCH OBJECTIVES (DEEP RESEARCH DIRECTIVES)
- **Derive Lattice-Locked RoPE Formulations:** Formulate a RoPE implementation that maps discrete lattice coordinates to absolute, distance-invariant frequency bases. Eliminate all sequence-length or max-distance normalization scaling to guarantee zero-shot cross-distance generalization.
- **Formulate Calibrated Physics Penalty:** Design the mathematical integration of $-\beta \cdot [-\log(p)] \cdot GraphDistance$ into the softmax attention mechanism. Prove it bounds the effective receptive field proportionally to MWPM probability decay.
- **Solve $D_4$ Equivariance in Attention:** Derive exact weight-sharing schemas and group-invariant projection bases (for $W_Q, W_K, W_V$) that make the attention mechanism strictly equivariant under $D_4$ spatial symmetry and time-reversal point inversion.
- **Design Spectral Attention Gating:** Propose an FFT-based or dual-stream attention modification to isolate high-frequency topological errors from low-frequency hardware leakage. Define the filter functions.