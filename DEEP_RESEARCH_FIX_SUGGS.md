# High-Performance Neural Decoding: Empirical Diagnostics & Prioritized Fixes

> **Implementation status (2026-04-17):**
> - FIX 1 (DIPE) — landed in `notebooks/model.py:build_rope_2_5d`.
> - FIX 2 — partial: hard `create_locality_mask` deleted; graph-adjacency replacement + soft `-log(p)` physics penalty pending.
> - FIX 3 (physics-invariance tests) — landed in `tests/test_equivariance.py`; 4/4 tests pass locally.
> - FIX 8 (D₄ TTA) — landed in `research_symmetries.py:get_d4_permutations` + `03_evaluation.ipynb` cell-14. Z₂×Z₂ subgroup (e, r2, eT, r2T) — 90° rotations rejected, swap X/Z stabilizer types in `rotated_memory_z`.
> - FIX 9 (stratified p-bin sampling) — subsumed by Cycle D on-the-fly pipeline: non-uniform p-schedule weighted $\propto 1/\sqrt{p}$ oversamples low-p bins during training.
> - FIX 10 (gradient checkpointing via `nn.remat`) — landed in `notebooks/model.py:302-312`.
> - FIX 4, 5, 6, 7, 11, 12 pending.
> - Eval-side infra added: Wilson 95% CI bands on LER plots, checkpoint version guard (accepts `'dipe-no-mask'` + `'dipe-no-mask-otf'`).
> - Infra: timestamped checkpoints, `requirements.txt` deps pin, on-the-fly STIM training pipeline with chunk-scan + prefetch.
> Completed sections commented out below; reopen if results regress.

## 1. Executive Diagnostic Summary
Transformer decoder (1.3M params) collapses as code distance ($d$) scales. Superior fidelity at $d=3$, no stable pseudo-threshold at higher distances.

### 1.1 Performance Metrics & Inversion Points
* **$d=3$:** Transformer beats MWPM, peak **~29.7%** gain at $p \approx 0.00531$.
* **$d=5$:** Inversion at $p=0.01$, Transformer loses to baseline (-0.87%).
* **$d=7$:** Catastrophic collapse. **-347.22%** deficit vs MWPM at $p=0.01$.

### 1.2 Threshold Analysis
| Decoder Architecture | Code Distance Crossing | Estimated Threshold ($p_{th}$) |
| :--- | :--- | :--- |
| MWPM (Baseline) | $d=5$ to $d=7$ | $\sim 0.0395$ |
| **Current Transformer** | **$d=5$ to $d=7$** | **$\sim 0.0131$** |

Collapse from $0.0269$ ($d=3 \to 5$) to $0.0131$ ($d=5 \to 7$) proves architecture cannot use redundant detectors — more qubits add more noise than param capacity and attention resolve.

---

## 2. Identified Technical Bottlenecks
<!-- [IMPLEMENTED 2026-04-17] Bottleneck #1 (RoPE Angular Drift) addressed by DIPE — see FIX 1 below.
1.  **RoPE Angular Drift:** Scaling coords by `code_distance` and `measurement_rounds` breaks translational invariance. Same error vector maps to different rotation frequencies at $d=3$ vs $d=7$, killing zero-shot extrapolation.
-->
2.  **Quadratic Sequence Scaling:** $\mathcal{O}(L^2)$ self-attention ($L \approx T \times d^2$) yields $\mathcal{O}(d^4)$ scaling, overwhelms TPU v6e VMEM, forces HBM swapping.
3.  **Combinatorial Capacity Limits:** 1.3M params insufficient for all-to-all interactions of 336+ detectors at $d=7$ under dense attention.
<!-- [PARTIAL 2026-04-17] Bottleneck #4: hard Manhattan mask deleted from `model.py`. Graph-adjacency replacement pending — see FIX 2.
4.  **Locality Mask Violates Physical Adjacency:** `create_locality_mask` applies single Manhattan budget across $(x, y, t)$, admits detector pairs sharing no qubit while excluding physically-adjacent pairs MWPM connects.
-->
5.  **CLS Information Bottleneck:** Single CLS token = sole classifier input, forces 336 detectors at $d=7$ through one 128-dim slot.

Note: All regimes constrained to $p < 0.02$ where state-of-art quantum processors operate. No fix extends training/eval past this band.

---

## 3. Prioritized Technical Fixes

### CRITICAL

<!-- [IMPLEMENTED 2026-04-17] FIX 1 (DIPE) landed.
     - `notebooks/model.py:build_rope_2_5d` now consumes raw integer detector coords; per-distance rescaling deleted.
     - RoPE base frequencies retuned to `base_spatial=100`, `base_temporal=20` for surface-code positional ranges.
     - `notebooks/02_model_and_training.ipynb` and `notebooks/03_evaluation.ipynb` `get_detector_coords` now origin-shift only (no normalize).
     - Existing checkpoints break; new model_version tag = 'dipe-no-mask'.

#### FIX 1: Distance-Invariant Position Encoding (DIPE)
**Target:** `model.py` → `build_rope_2_5d`
* **Action:** Kill RoPE angular drift — remove all dynamic dependencies on global sequence length or code distance.
* **Implementation:** Anchor rotation matrix $\mathbf{R}_{\Theta}$ basis frequencies $\theta_k = \beta^{-2k/D_h}$ to exact integer lattice coords $(dx, dy)$. Delete `coords[:,0] * code_distance` rescaling in `build_rope_2_5d`; feed integer lattice positions directly.
* **Goal:** Restore translational invariance so network learns error propagation physics, not lattice boundaries. Prerequisite for FIX 7.
-->

<!-- [PARTIAL 2026-04-17] FIX 2 first action (mask removal) landed; graph-adjacency replacement and soft `-log(p)` bias pending.
     - `create_locality_mask` deleted from `notebooks/model.py`.
     - `mask` parameter dropped from `TransformerBlockWithRoPE.__call__`; mask plumbing in `TransformerQEC.__call__` removed.
     - The `-10^9` hard-gate path is gone; attention is fully dense again, restoring the maskless-baseline behaviour that historically beat MWPM at d=5.
     - Still pending (queued for next cycle, gated on d=5 recovery): exact-adjacency graph mask from `circuit.detector_error_model()` + soft physics penalty `score -= β · (-log p) · d_graph(i,j)`. JAX Pallas sparse kernels still in scope.

#### FIX 2: Graph-Based Locality Mask with Soft Physics Penalty
**Target:** `model.py` → `create_locality_mask` (line 138) + `TransformerBlockWithRoPE` (line 214)
* **Observation:** Current mask applies *single* Manhattan budget across $(x, y, t)$:
  ```python
  manhattan_dist = jnp.sum(jnp.abs(coords_q - coords_k), axis=-1)
  return jnp.expand_dims(manhattan_dist <= radius, axis=(0, 1))
  ```
  Two independent defects:
    1. **Anisotropy violation** — `radius=2` lets detector reach neighbor 2 spatial steps away **or** 2 rounds away, but *not both*. Contradicts (2+1)D RoPE design.
    2. **Physical-adjacency mismatch** — even with corrected anisotropic box, Manhattan ball admits pairs sharing no qubit and excludes boundary-adjacent pairs MWPM connects.
* **Action:** Replace `create_locality_mask` with exact syndrome-graph adjacency matrix $\mathbf{M}$ from STIM detector error model. Soften hard $-10^9$ gate into continuous, $p$-adaptive distance bias.
* **Implementation:**
    * **Graph extraction:** Pre-compute $\mathbf{M}$ from `circuit.detector_error_model()` — edges exist iff single elementary error flips both detectors. Cache per $(d, \text{rounds})$. Naturally encodes spatial edges (shared stabilizer) and temporal edges (same detector across rounds) as distinct types, subsuming axis-decoupling.
    * **Hard gate (FLOP reduction):** Restrict attention to $j \in \mathcal{N}(i)$ via cached adjacency. Deploy **JAX Pallas kernels** so MXU skips memory loads for masked blocks. Surface codes yield ~6 neighbors/detector, converting dense $\mathcal{O}(L^2)$ to sparse $\mathcal{O}(L)$.
    * **Soft bias (physics-calibrated):** Within kept edges, subtract continuous penalty from pre-softmax score:
    $$\text{AttentionScore}_{ij} \mathrel{-}= \beta \cdot [-\log p] \cdot d_{\text{graph}}(i, j)$$
    $\beta$ learned, $d_{\text{graph}}$ = shortest-path graph distance. Low $p$ → sharp penalty enforcing strict locality; near threshold → relaxes automatically, matching MWPM probability decay.
* **Goal:** Replace heuristic locality with *exact* physical adjacency from baseline decoder. Translate sparsity into hardware FLOP reductions (supporting $<1\mu s$ latency target), give receptive field $p$-adaptive shape.
* **Why CRITICAL:** Single fix addresses two bottlenecks (§2 items 2 and 4) and corrects physics violation in current mask.
-->

<!-- [IMPLEMENTED 2026-04-17] FIX 3 landed.
     - `.gitignore` line `tests/` removed.
     - New `tests/test_equivariance.py`: parametrized over d ∈ {3, 5}, verifies rot180 + time-reversal permutation invariance at random init in float32 (atol=1e-4).
     - 4/4 tests pass locally in 13.7s (CPU, no TPU needed).
     - Translation-by-integer test deferred — surface code detectors are not lattice-translation-symmetric (boundary detectors don't translate to other valid detectors).

#### FIX 3: Physics-Invariance Regression Tests
**Target:** `tests/test_notebooks_compat.py`
* **Observation:** Current suite checks shapes, NaN, checkpoint round-trips — zero physical invariants. Any regression in FIX 1/2 passes CI unnoticed.
* **Action:** Add Phase 10 tests:
  1. Generate syndrome + coords at $d=5$, apply $D_4$ group element via `research_symmetries.py:get_rot180_permutation`, assert `logits[orig] ≈ logits[rotated]` within floating tolerance.
  2. Translate all detector coords by integer vector, assert output logits unchanged (tests DIPE).
  3. Apply temporal reversal via `get_time_reversal_permutation`, assert equivariance.
* **Why CRITICAL:** **Prerequisite gate** for FIX 1 and FIX 2. Without these tests, no way to trust DIPE is distance-invariant or graph masking preserves equivariance.
-->

---

### HIGH

#### FIX 4: Attention Pooling Head (Eliminate CLS Bottleneck)
**Target:** `model.py:TransformerQEC.__call__` (line 308, `x[:, 0]`)
* **Observation:** Classification reads one token — prepended CLS — after 4 layers. 336 detectors at $d=7$ forced through single 128-dim slot = info bottleneck worsening monotonically with $d$. CLS inserted with identity rotation (`cls_cos=1, cls_sin=0`), so final-layer attention from CLS has *no* positional discrimination over detectors.
* **Action:** Replace CLS read-out with learned-query attention pooling:
  $$h = \mathrm{softmax}\!\left(\frac{q_{\text{pool}} K^\top}{\sqrt{d_h}}\right) V$$
  $q_{\text{pool}} \in \mathbb{R}^{d_h}$ = single learned query, $K, V$ = projections of all detector tokens. Adds $\sim 3 d_{\text{model}}^2$ params (negligible), removes CLS token entirely.
* **Goal:** Lift $d=7$ bottleneck without growing depth or width.

#### FIX 5: Qubit-Centric Projection (QCT)
**Target:** `model.py` → Input Embedding Layer
* **Action:** Deprecate naive binary syndrome embedding.
* **Implementation:** Concatenate adjacent $X$ and $Z$ stabilizer events into unified physical qubit features: $\phi_i = \mathbf{W}_e \xi_i + \mathbf{b}_e$.
* **Goal:** Halve spatial sequence length $L$, cut $\mathcal{O}(d^4)$ bottleneck, focus param capacity on physical qubit space.

#### FIX 6: MWPM Soft-Label Distillation
**Target:** Loss function in `02_model_and_training.ipynb`
* **Observation:** Focal loss ($\gamma=2, \alpha=0.75$) partly counters class imbalance, but at $p=0.002$ positive rate is $\sim 10^{-4}$ — most gradients from trivially-decoded negatives. PyMatching exposes per-sample log-odds = *soft* teacher encoding where MWPM confident vs uncertain.
* **Action:** Augment loss with KL term against MWPM posterior:
  $$\mathcal{L} = \lambda \mathcal{L}_{\text{focal}}(y_{\text{true}}) + (1 - \lambda)\, \mathrm{KL}\!\left(\sigma(\text{logits}) \,\|\, \sigma(\ell_{\text{MWPM}})\right)$$
  $\lambda \in [0.5, 0.8]$ annealed downward across training.
* **Rationale:** Teaches transformer to *reproduce* MWPM where right, frees remaining capacity for correlated-error structure ($Y$-errors, hook errors) where MWPM wrong — preserves $d=3, 5$ advantage into $d=7$.

#### FIX 7: Unified Multi-Distance Training with Shared Backbone
**Target:** Training loop + model instantiation (depends on FIX 1)
* **Observation:** `results/` has three independent checkpoints (`transformer_qec_d3.pkl`, `_d5.pkl`, `_d7.pkl`). Each distance pays full param budget in isolation. $d=7$ model sees ~10M syndromes; unified backbone sees 30M+ across shared topological features.
* **Action:** After FIX 1 restores distance-invariance, train **single** weight set on batches interleaved from $d \in \{3, 5, 7, 9\}$ with per-sample detector coords. Static pad to $L_{\max} = L(d=9)$ + attention masks over pad tokens (compatible with FIX 2).
* **Goal:** Convert 3-checkpoint regime into unified decoder with 4× effective training set at $d=7$, *zero* extra params. Biggest free lunch once FIX 1 lands.

<!-- [IMPLEMENTED 2026-04-17] FIX 8 landed (Z₂ × Z₂ subgroup, not full D₄).
     - `research_symmetries.py:get_d4_permutations` enumerates 8 spatial × 2 time = 16 candidates and filters to valid bijective detector permutations.
     - For `rotated_memory_z` only `{e, r2, eT, r2T}` survive — 90°/270° rotations and diagonal reflections swap X/Z stabilizer types and are correctly rejected by the bijectivity check.
     - `03_evaluation.ipynb` cell-14: vmap'd softmax-average across surviving group elements, stored in `results_tta` with Wilson CIs. Combined plot (cell-16) auto-overlays `Transformer+TTA d={d}` curve when `results_tta` is defined.
     - Expected ~1-3% LER reduction; latency ~G× = 4× (acceptable for offline eval).

#### FIX 8: D₄ Test-Time Augmentation (TTA)
**Target:** Evaluation pipeline (`03_evaluation.ipynb`)
* **Observation:** `research_symmetries.py` verifies $180°$ rotation and time-reversal are valid detector permutations, but result unused.
* **Action:** At inference, apply all 8 $D_4$ group elements to `(syndrome, coords)`, forward pass each, average logits in probability space. **Zero parameters**; 8× latency acceptable — evaluation is offline.
* **Expected Impact:** ~1–3% absolute LER reduction at $d=5, 7$ per equivariant-ensemble literature — directly measurable against existing CSV.
-->

> **⚠ Post-landing caveat (2026-04-18, code change deferred):** the 4-element `{e, r2, eT, r2T}` group shipped in FIX 8 includes time-reversal, but the `rotated_memory_z` circuit is **boundary-asymmetric in time** — `|0⟩` init is noiseless while the final data-qubit measurement applies `before_measure_flip_probability=p`. The `t=0` and `t=rounds` boundary detectors therefore carry different noise budgets, so `T: t → tmax - (t - tmin)` is only an *approximate* symmetry of the syndrome marginal. Averaging softmax probabilities over the `*T` elements mixes in a non-symmetry term that biases the TTA output.
>
> **Next TTA iteration** (gated on the current 4-element eval run locking first, so both variants can be compared against the same checkpoint): flip `03_evaluation.ipynb` cell-14 to `get_d4_permutations(coords_np, include_time_reversal=False)` → surviving group is the spatial Klein-2 `{e, r2}`, which **is** an exact syndrome-distribution symmetry (180° rotation about the patch center maps `rotated_memory_z` detector lattice onto itself and preserves the $Z_L$ observable). Update cell-13 markdown accordingly. Keep the `include_time_reversal` kwarg in `research_symmetries.py` — genuinely useful once circuit-level noise models with time-symmetric noise channels are explored.

---

### MEDIUM

<!-- [IMPLEMENTED 2026-04-17 Cycle D — subsumed by on-the-fly pipeline.]
     - Non-uniform p-schedule in `02_model_and_training.ipynb` cell-14 weights step-to-p assignment as $w_i \propto 1/\sqrt{p_i}$.
     - Low-p bins are oversampled in the step sequence, so label-1 positives per batch are closer to uniform across p without explicit oversample + reweight.
     - No explicit focal-loss reweighting needed — the oversampling is in the data stream, not the loss.
     - Full class-balanced oversample (positive share >= 25%) NOT done; deferred if training-time loss still shows vanishing low-p gradient signal.

#### FIX 9: Stratified Class-Balanced Sampling Per p-Bin
**Target:** `02_model_and_training.ipynb:generate_dataset`
* **Observation:** `generate_dataset(d, p_values, shots_per_p)` emits equal shots per $p$. For $p \le 0.003$ batches are $\gtrsim 99.9\%$ label-0, so optimizer sees vanishing positive-class gradient exactly where decoder advantage over MWPM largest.
* **Action:** Within each $p$-bin, oversample label-1 until positive share $\ge 25\%$ per batch. Reweight focal loss by inverse oversampling ratio to preserve unbiased risk estimate.
* **Cost:** 2–3× more STIM sampling per low-$p$ bin; zero model changes.
-->

<!-- [IMPLEMENTED 2026-04-17 Cycle C] FIX 10 landed.
     - `notebooks/model.py:302-312`: `block_cls = nn.remat(TransformerBlockWithRoPE)` hoisted outside the layer loop; layer loop instantiates `block_cls` instead of the raw class.
     - Forward output bit-identical; existing `dipe-no-mask` checkpoints load unchanged.
     - Equivariance tests pass 4/4 in 12.15s post-remat (architectural invariants preserved).
     - Unlocks d=9 training headroom at 1.3M params.

#### FIX 10: Gradient Checkpointing (`nn.remat`) on Transformer Blocks
**Target:** `model.py:TransformerQEC.__call__` layer loop (line 303)
* **Observation:** At $d=7$, each dense attention head holds $L^2 = 336^2 \approx 113\text{k}$ floats/layer. 4 heads × 4 layers × `bf16` → live activation footprint dominates VMEM, silently forces XLA to HBM-spill.
* **Action:** Wrap block with `nn.remat(TransformerBlockWithRoPE, ...)`. Costs ~30% extra compute, yields $O(1/\sqrt{N_{\text{layers}}})$ activation memory — enables $d=9$ training at current 1.3M param budget.
-->

#### FIX 12: Hybrid RoPE + Reflection-Symmetric Additive Attention Bias
**Target:** `model.py` → `TransformerBlockWithRoPE.__call__` (pre-softmax, line ~196)
* **Observation:** Current attention uses 2.5D RoPE only. Per-axis score decomposes as
  $$S_{\text{axis}}(i,j) = \sum_k \big[A_k \cos(\theta_k \Delta) + B_k \sin(\theta_k \Delta)\big]$$
  where $A_k = q_a^{(i)} k_a^{(j)} + q_b^{(i)} k_b^{(j)}$ (even, reflection-symmetric) and $B_k = q_a^{(i)} k_b^{(j)} - q_b^{(i)} k_a^{(j)}$ (odd, breaks reflection). Reflection symmetry is **not enforced** — sin coefficients $B_k$ are free DOF the model must learn to suppress (or use, depending on noise model). Under Colab compute, this costs sample efficiency wherever the symmetry is approximately physical.
* **Action:** Add a small learnable additive bias $B_{\text{sym}}(|\Delta x|, |\Delta y|, |\Delta t|)$ to attention logits pre-softmax. Bias is **content-blind** and **even in each axis** → bakes in $\mathbb{Z}_2 \times \mathbb{Z}_2$ reflection-symmetric inductive prior + soft locality. RoPE retained intact; sin DOF kept so model can still learn axis asymmetry where data demands.
* **Implementation:**
    * Lookup table indexed by $(|\Delta x|, |\Delta y|, |\Delta t|)$ bins. Max range $\sim 2(d-1)$ spatial $\times$ $d$ temporal → $\sim 5\times 5\times 5 = 125$ floats per head at $d=5$. Trivial param overhead.
    * Per-head learnable scale $m_h$ (ALiBi-style) → heads learn distinct decay rates, multi-scale locality.
    * Init: small negative slope $B_{\text{sym}}(\Delta) = -m \cdot (|\Delta x| + |\Delta y| + \alpha |\Delta t|)$ to seed locality prior; refined via gradient.
    * Insertion point: add to attention logits **before softmax**:
      $$S(i,j) = \frac{\langle R_{xy}(\Delta x, \Delta y) \mathbf{q}^{(xy)}_i, \mathbf{k}^{(xy)}_j\rangle_{d_s} + \langle R_t(\Delta t) \mathbf{q}^{(t)}_i, \mathbf{k}^{(t)}_j\rangle_{d_t}}{\sqrt{d_h}} + B_{\text{sym}}(|\Delta x|, |\Delta y|, |\Delta t|)$$
    * Optional: full $D_4$ (add 90° rotation symmetry) via $B_{\text{sym}}(\min(|\Delta x|, |\Delta y|), \max(|\Delta x|, |\Delta y|), |\Delta t|)$ — only if ablation supports under chosen noise model.
    * MLP form (small 2-layer net on $(|\Delta x|, |\Delta y|, |\Delta t|)$) preferred over raw lookup if cross-distance generalization (FIX 7) target — table is bounded by max trained index; MLP is smooth.
* **Why hybrid (not pure even kernel):** additive bias **does not cancel the sin term** — even and odd functions are orthogonal under reflection. Bias provides a *soft* prior toward reflection symmetry; model retains sin DOF to learn real asymmetric signal. This is intentional — see noise-model applicability below.
* **Noise-model applicability:**
    * **Code-capacity / depolarizing:** $D_4$ exact. Bias = strong regularizer, near-zero cost; sin DOF idle.
    * **Phenomenological (current synthetic STIM):** $D_4$ approximately exact in bulk; `rotated_memory_z` boundary-asymmetric in time (noiseless $|0\rangle$ init vs. noisy final measurement). Bias acts as exact prior in bulk + soft prior at boundary. Cheap regularizer, low risk.
    * **Circuit-level (next data target):** $D_4$ **broken** by CNOT scheduling order (hook errors tilt one diagonal), X/Z stabilizer asymmetry, idle errors during measurement. Bias = inductive prior toward symmetric solution; RoPE sin DOF lets model learn residual hook-tilt asymmetry. Hybrid is the **correct baseline** — pure even kernel would erase real signal.
    * **Real hardware (Sycamore/Willow):** $D_4$ further broken by per-site $T_1, T_2$ variation, gate fidelity heterogeneity, leakage, asymmetric readout. Bias still useful as locality prior but **must be paired with per-site calibration features** as separate token augmentation (out of scope for this fix).
* **Why HIGH (not CRITICAL):** does not address a #1 bottleneck — RoPE already lands relative-distance signal cleanly post-DIPE; this fix *strengthens* the symmetry/locality prior rather than correcting a bug. Land after FIX 4/5/6, before FIX 7 (helps unified-distance training generalize).
* **Compatibility:**
    * Soft (gradient flows everywhere) → does not repeat the d=5 hard-mask regression (memory: hard masks broke d=5; soft biases are safe).
    * Decouples cleanly from FIX 2's graph-adjacency mask + $-\log p$ physics penalty — those operate on graph-distance scaling; this operates on Manhattan-distance reflection symmetry. Both can co-exist additively.
    * Bumps `model_version` tag (e.g. `'dipe-no-mask-otf-symbias'`); existing checkpoints incompatible (extra params).
* **Test gate:** add to `tests/test_equivariance.py`: with $B_{\text{sym}}$ enabled at random init, the bias contribution alone should be exactly invariant under axis reflection ($\Delta x \to -\Delta x$); full model still tested under existing rot180 / time-reversal permutation invariants.

##### Design Discussion Summary (2026-04-18)

**Current RoPE score structure.** Spatial chunks in `build_rope_2_5d` use **interleaved** per-axis pairs (not joint 2D rotation). Each pair rotates purely on $x$ or purely on $y$. $Q \cdot K^\top$ decomposes additively:
$$S(i,j) = \sum_{\bullet \in \{x,y,t\}} \sum_k \big[A_k^{(\bullet)} \cos(\theta_k^{(\bullet)} \Delta_\bullet) + B_k^{(\bullet)} \sin(\theta_k^{(\bullet)} \Delta_\bullet)\big]$$
Separable in $x, y, t$. No cross terms. Reflection symmetry = killing $B_k^{(\bullet)}$ (odd). Impossible via additive even bias (even ⊥ odd under reflection).

**Why separate-axis RoPE beats Manhattan-scalar.** Manhattan collapses direction; X/Z stabilizers live on perpendicular sublattices, chains direction-dependent. Separate axes preserve $2\times$ info bandwidth per pair. Circuit-level noise breaks isotropy (hook tilt, CNOT-order asymmetry) → need per-axis DOF, not tied rotation.

**3:1 spatial:temporal dim ratio justified by physics.** Spatial range $\sim 2(d-1)$ (~8 at d=5), temporal range $\sim d$ (~5). Info-theoretic floor ~3:2; 3:1 gives margin. 1:1 wastes temporal capacity on freq bands exceeding $\Delta t$ range (same failure mode as NLP base-10000 default — zero-signal bands). 5:1+ under-resolves measurement-error temporal correlations.

**Physical bound on correlation span.** Single errors → adjacent detectors. Hook errors → 2 diagonal steps. Measurement → 1 round. Max meaningful pair distance bounded by chain length ≤ `d/2` spatial, ~2 temporal. Far-pair correlations only exist under cosmic-ray / burst events — out-of-distribution, handled at device level, not decoder job. **Locality prior is physics-exact for the training distribution**, not an approximation.

**`B_sym(|Δx|,|Δy|,|Δt|)` vs FIX 2's `−β·log(p)·d_graph(i,j)` — complementary, not redundant.**

| Property | `B_sym` | `−log(p)·d_graph` |
|---|---|---|
| Distance metric | Manhattan (geometric) | syndrome-graph shortest path (topological) |
| Symmetry encoded | $\mathbb{Z}_2 \times \mathbb{Z}_2$ reflection | inherits surviving subgroup of physics |
| `p`-adaptive | ✗ | ✓ |
| Boundary-aware | ✗ | ✓ |
| Hook-tilt aware | ✗ | ✓ |
| Precompute | none | DEM extract + all-pairs BFS, cached per `(d, noise)` |
| Params | ~125 floats/head + `m_h` | `β` per head |

Positively rank-correlated in bulk (~0.7+), conflict only on hook-tilt diagonals / boundary / X-Z type splits. Softmax-invariance under shared additive offset means `B_sym` cannot corrupt `−log(p)·d_graph`'s relative signal; at worst mildly mis-regularizes.

**Per-`p` regime split (both active, circuit-level):**
- Low `p`: `−log(p)·d_graph` saturates attention onto graph-neighbors → dominates. `B_sym` smooths within-neighbor ranking.
- Near threshold: `−log(p)·d_graph` relaxes by design → `B_sym` carries locality. Complementary across `p` range; no per-`p` retraining needed.

**Latency impact — negligible to net positive.**
- Per-step: ~2–5% FWD/BWD overhead (elementwise add on `L²` logit matrix, caches amortized).
- Convergence: expected ~10–20% fewer epochs. Biases seed locality → Q/K capacity redirected to content pathway.
- Memory: ~1.8 MB `d_graph` for d=7, ~500 floats `B_sym` table. Zero per-batch.
- No XLA recompilation; no precision issues (softmax in f32 preserved).

**Vanishing-gradient risks — real but monitorable, not structural.**
1. **Softmax saturation from unchecked scale growth.** Mitigate: init `m_h ~ 0.1`, `β ~ 0.05`; weight decay; optional `softplus` parameterization.
2. **Sin-DOF starvation** — quiet failure where biases subsume attention shape, leaving hook-tilt asymmetry unlearned. Monitor `‖sin coeffs‖ / ‖cos coeffs‖` per head per epoch; drop `B_sym` if ratio → 0.
3. Rare-positive gradient dilution exists in baseline too — biases don't worsen it.
4. Classic deep-net vanishing N/A (4 pre-norm layers, RoPE norm-preserving, `nn.remat` exact).

**Ablation order (d=5 Colab cycles):**
1. `B_sym` alone (cheapest, tests symmetry+locality hypothesis).
2. `−log(p)·d_graph` alone (FIX 2, queued).
3. Both combined.
4. If `−log(p)·d_graph` subsumes `B_sym` under circuit-level → drop `B_sym` (Occam).

**Combined final form:**
$$S(i,j) = \frac{\langle R_{xy}(\Delta x, \Delta y) \mathbf{q}_i^{(xy)}, \mathbf{k}_j^{(xy)}\rangle_{d_s} + \langle R_t(\Delta t) \mathbf{q}_i^{(t)}, \mathbf{k}_j^{(t)}\rangle_{d_t}}{\sqrt{d_h}} + B_{\text{sym}}(|\Delta x|, |\Delta y|, |\Delta t|) - \beta \cdot [-\log p] \cdot d_{\text{graph}}(i, j)$$
RoPE: content-modulated relative position. `B_sym`: symmetric geometric prior. `−log(p)·d_graph`: physics-aware `p`-adaptive adjacency prior. Three channels, orthogonal roles.

**Soft over hard, always.** Soft biases preserve gradient flow + leave model able to override under anomalous (cosmic-ray-like) shots if training data later demands. Hard masks (removed 2026-04-17 after d=5 regression) are disallowed going forward — lesson logged in memory.

---

#### FIX 11: Mamba-Augmented Temporal Routing
**Target:** Temporal Dimension Processing
* **Action:** Replace temporal self-attention with Selective State-Space Model (SSM).
* **Implementation:** Route spatially-attended features through Mamba block for linear $\mathcal{O}(d^2)$ temporal complexity.
* **Goal:** Enable infinite syndrome stream processing without global temporal attention memory overhead.
* **Why MEDIUM:** Highest-risk rewrite — speculative, land only after cheaper fixes stabilize.

---

## 4. Hardware Alignment (TPU v6e)
For $10\times$ perf gain, enforce XLA optimizations:
1.  **Precision Discipline:** Q-K matmuls in `bfloat16`, upcast to `float32` for `softmax`, back to `bfloat16` for value aggregation.
2.  **Static Compilation:** Pad all windows to constant size — block XLA recompilation during real-time decoding.
3.  **VMEM Management:** Use Pallas to orchestrate DMA transfers HBM→VMEM, stop dense attention matrices from thrashing memory hierarchy.

---

## 5. Landing Order

Fixes not fully independent — wrong order wastes experiments:

1. ~~**FIX 3** (physics-invariance tests) — must exist *before* FIX 1/2, or neither trusted.~~ **[IMPLEMENTED 2026-04-17. 4/4 tests pass locally.]**
2. ~~**FIX 1** (DIPE) — headline architectural fix.~~ **[IMPLEMENTED 2026-04-17.]**
3. **FIX 2** (graph masking + soft physics penalty) — parallel with FIX 1; disjoint attention block parts. **[PARTIAL — mask removed; graph mask + soft bias queued for next cycle.]**
4. **FIX 4** (attention pooling) — independent of FIX 1/2; parallel.
5. **FIX 5** (QCT) — sequence-length reduction; compatible with landed FIX 1/2.
6. **FIX 6** (MWPM distillation) — largest expected $d=7$ impact after FIX 1.
7. **FIX 7** (unified multi-distance training) — depends on FIX 1. **(Now unblocked.)**
8. ~~**FIX 8** (TTA) — evaluation-only; land after training changes stabilize.~~ **[IMPLEMENTED 2026-04-17 — Z₂×Z₂ subgroup, 90°/270°/diagonals correctly auto-rejected for `rotated_memory_z`.]**
9. ~~**FIX 9** (stratified p-bin sampling)~~ **[IMPLEMENTED 2026-04-17 Cycle D — subsumed by on-the-fly pipeline's $1/\sqrt{p}$ weighted schedule.]** ~~**FIX 10** (gradient checkpointing)~~ **[IMPLEMENTED 2026-04-17 Cycle C.]**
10. **FIX 12** (hybrid RoPE + reflection-symmetric additive bias) — symmetry/locality inductive prior; cheap, safe (soft), compatible with FIX 2 and FIX 7. Land after FIX 4/5/6, before FIX 7.
11. **FIX 11** (Mamba) — last, highest-risk rewrite.

**Eval-side scaffolding (added outside original FIX list):**
- ~~Wilson 95% CI bands on LER plots~~ **[IMPLEMENTED 2026-04-17 Cycle B.]**
- ~~Checkpoint version guard rejecting legacy pkls~~ **[IMPLEMENTED 2026-04-17 Cycle B; Cycle D: accepts both `'dipe-no-mask'` and `'dipe-no-mask-otf'`.]**
- ~~Timestamped checkpoint filenames~~ **[IMPLEMENTED 2026-04-17 Cycle C.]**
- ~~`requirements.txt` dependency pinning~~ **[IMPLEMENTED 2026-04-17 Cycle D.]**
- ~~On-the-fly STIM training data pipeline~~ **[IMPLEMENTED 2026-04-17 Cycle D.]**