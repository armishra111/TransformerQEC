# Project State
# Last Updated: 2026-04-17

## Current Status
- **Core Research Completed:** Transformer encoder trained for code distances $d \in \{3, 5, 7\}$ on rotated surface codes. Original masked checkpoints beat MWPM for $d=3$ but regressed at $d=5,7$. Pre-mask 1.3M-param model historically beat MWPM at $d=5$.
- **Constraint:** Code refactoring + implementation now active under Claude Code. Doc-only role for Gemini.
- **Core Directive:** Parameter efficiency + elegant architectural constraints (inductive biases) over brute-force scaling.

## Recent Implementation — Cycle A (2026-04-17, architecture fixes)
- **Train/eval architecture unified.** `notebooks/02_model_and_training.ipynb` cell-7 inline `TransformerQEC` deleted; replaced with `from model import TransformerQEC` after Drive-mount. Single source of truth in `notebooks/model.py`. Eliminates the architecture drift that caused the same checkpoint to use different RoPE math + a phantom mask between train and eval.
- **Locality mask removed.** `create_locality_mask` deleted; `mask` parameter dropped from `TransformerBlockWithRoPE.__call__`; mask construction in `TransformerQEC.__call__` removed. User-confirmed root cause for the d=5 regression.
- **DIPE landed.** `build_rope_2_5d` now consumes raw integer detector coords (origin-shifted only). Per-distance rescaling (`coords * code_distance` / `coords * seq_len`) deleted. RoPE base frequencies retuned: `base_spatial=100`, `base_temporal=20`. Same physical lattice cell yields identical RoPE angle across $d$. `code_distance`/`measurement_rounds` retained as `nn.Module` fields for legacy checkpoint config-dict compatibility but unused at runtime.
- **Checkpoint break.** Existing `transformer_qec_d{3,5,7}.pkl` no longer load (Q/K projections trained against the old RoPE geometry). New checkpoints tagged `model_version: 'dipe-no-mask'`.
- **Verification gate (pending Colab run):** `02_model_and_training.ipynb` `DISTANCE = 5`. Single d=5 retrain → eval. Acceptance: d=5 Transformer LER ≤ MWPM at p ∈ [0.005, 0.01]. If yes, next cycle adds soft `-log(p) · graph_distance` attention bias.

## Recent Implementation — Cycle B (2026-04-17, eval-side scaffolding)
Orthogonal to Cycle A (no changes to training notebook or `model.py` math), so d=5 Colab signal stays clean.
- **Wilson 95% CI bands.** `03_evaluation.ipynb` cell-12 computes Wilson CI per (d, p) for both decoders via new `wilson_ci(successes, total, z=1.96)` helper. Per-decoder (cell-15) and combined (cell-16) plots shade `fill_between` bands. CSV (cell-18) gains `mwpm_ci_lo/hi`, `tf_ci_lo/hi` columns. Makes low-p comparisons interpretable (at p=0.001 with NUM_TEST=100k, expected positives ~10, Poisson noise massive).
- **Checkpoint version guard.** cell-9 globs `best_d*_checkpoint.pkl` (new training-side filename). cell-10 asserts `ckpt['config']['model_version'] == 'dipe-no-mask'`; legacy files skip with warning instead of silently loading and producing nonsense LERs.
- **Equivariance regression tests (FIX 3).** `.gitignore` line `tests/` removed. New `tests/test_equivariance.py` parametrized over d ∈ {3, 5}, verifying permutation invariance under `get_rot180_permutation` and `get_time_reversal_permutation` at random init, float32, atol=1e-4. **4/4 tests pass locally in 13.7s.** `research_symmetries.py` bottom sweep wrapped in `if __name__ == "__main__":` so imports are silent.
- **D₄ × T test-time augmentation (FIX 8).** New `get_d4_permutations(coords, include_time_reversal=True)` in `research_symmetries.py`: enumerates 16 candidate transforms (8 spatial × 2 time), validates each via nearest-neighbor detector match + bijectivity check, returns only the surviving subset. For `rotated_memory_z` at d ∈ {3, 5, 7} the survivors are `{e, r2, eT, r2T}` — 90°/270° rotations and diagonal reflections correctly rejected because they swap X/Z stabilizer types. New cell-14 in `03_evaluation.ipynb` averages softmax probabilities over surviving elements via `jax.vmap`, stores into `results_tta` with its own Wilson CIs. Combined plot (cell-16) auto-overlays `Transformer+TTA d={d}` curves when `results_tta` is defined. Expected ~1-3% LER reduction; ~4× latency is fine for offline eval.
  - **⚠ Caveat (documented 2026-04-18, code change deferred):** `rotated_memory_z` is NOT time-symmetric at its boundaries — `|0⟩` init is noiseless while the final data-qubit measurement carries `before_measure_flip_probability=p`. `T` is therefore only an *approximate* symmetry of the syndrome marginal; the `eT` / `r2T` elements in the current 4-member group average over a non-symmetry and biases the softmax mean. Planned fix: `include_time_reversal=False` → Klein-2 `{e, r2}` (exact symmetry). Deferred until the current 4-element run's results are saved so the two variants can be directly compared.

## Recent Implementation — Cycle C (2026-04-17, pre-run hygiene + FIX 10)
- **FIX 10: Gradient checkpointing.** `model.py:302-312`: `block_cls = nn.remat(TransformerBlockWithRoPE)` wraps each transformer block so activations are recomputed in backward pass. ~30% extra compute, ~$\mathcal{O}(1/\sqrt{N_{\text{layers}}})$ activation memory. Forward output bit-identical. Equivariance tests still pass 4/4 (12.15s post-remat).
- **Timestamped checkpoints.** Training `RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')` at cell-13; cell-14 writes `best_d{d}_{RUN_STAMP}_checkpoint.pkl` / `safety_d{d}_{RUN_STAMP}_epoch{N}.pkl`; config dicts carry the stamp. Eval cell-9 regex-parses + groups by d; cell-10 picks latest, supports `CHECKPOINT_OVERRIDES` for pinning.
- **Legacy pkl archive.** `results/transformer_qec_d{3,5,7}.pkl` → `results/legacy/` (git renames staged).
- **Minor hygiene.** research_symmetries variable-length coord guard; dup `drive.mount()` removed; eval cell-0 + training cell-5 markdowns updated.

## Recent Implementation — Cycle D (2026-04-17, on-the-fly STIM pipeline)
Bundled with Cycle C, applies to the pending d=5 Colab run. Training data now generated during training instead of materialized upfront.
- **`requirements.txt`** at repo root pins `stim>=1.14,<2`, `pymatching>=2`, `flax>=0.8`, `jax[tpu]>=0.4`, `optax>=0.2`, `numpy<2`. Both notebooks' cell-1 installs from it.
- **`02_model_and_training.ipynb` cell-4**: `build_samplers(d, p_values)` caches one `stim.Circuit.compile_detector_sampler()` per `p`. `generate_val_dataset` keeps bulk-materialized val (stable, deterministic seed).
- **cell-8**: `train_chunk` is `jax.lax.scan` over `CHUNK_SIZE=100` pre-sampled batches per TPU call — retains scan amortization. `eval_epoch` unchanged (scan-based, val bulk).
- **cell-14**: `ThreadPoolExecutor` (max_workers=1) prefetches chunk N+1 while TPU trains chunk N. Non-uniform p-schedule weighted by $1/\sqrt{p}$ oversamples low-p bins (subsumes FIX 9). Step-rate + recent-loss log every 100 steps. Checkpoints save under `'model_version': 'dipe-no-mask-otf'`.
- **`03_evaluation.ipynb` cell-10**: `EXPECTED_MODEL_VERSIONS = {'dipe-no-mask', 'dipe-no-mask-otf'}` accepts both regimes.

## Research Tracks (Impact Priority)
1. **~~Distance-Invariant Position Encoding (DIPE)~~ — IMPLEMENTED 2026-04-17.** Anchored RoPE to raw integer lattice coords. Eliminates angular drift. Unified multi-$d$ training still pending.
2. **Sequence Reduction & Hardware-Aware Masking:**
   - Hard locality mask **removed** 2026-04-17.
   - Soft Calibrated Physics Penalty $AttentionScore - \beta \cdot [-\log p] \cdot d_{graph}$ queued for next cycle (gated on DIPE d=5 result).
   - Qubit-Centric Projection (QCT) and JAX Pallas graph masking still pending.
3. **Symmetry-Aware Decoding (Physics-Inspired):**
   - ~~Test-Time Augmentation~~ **IMPLEMENTED 2026-04-17** (`get_d4_permutations` + vmap'd softmax-average, Z₂×Z₂ physical subgroup).
   - ~~Physics-invariance regression tests~~ **IMPLEMENTED 2026-04-17** (`tests/test_equivariance.py`, 4/4 passing).
   - In-model $D_4$ equivariant layers with weight sharing — still pending.
4. **Temporal Processing (Exploratory):** Mamba-Augmented Temporal Routing (SSMs) for linear temporal complexity. Spectral/Fourier Attention Gating to decouple hardware noise from topological errors. Pending.
5. **Hardware Alignment (TPU v6e):** bfloat16 matmul precision discipline ✓, **gradient checkpointing via `nn.remat` ✓ (2026-04-17 Cycle C)**, static XLA padding, VMEM DMA orchestration via Pallas, throughput profiling.

## Eval-Side Infrastructure (Cycle B)
- `03_evaluation.ipynb` now emits 95% Wilson CIs on every LER point (plots, CSV).
- Eval refuses to load legacy checkpoints — bumped model version guards against silent regressions.
- TTA overlay is opt-in (runs after base eval); no interference with the primary Transformer vs MWPM comparison.

## Next Steps
- [ ] **Verify Cycle A + B on Colab:** retrain d=5, run `03_evaluation.ipynb` end-to-end (base eval → TTA eval → plots). Confirm Wilson-CI bands show d=5 Transformer LER ≤ MWPM at p ∈ [0.005, 0.01] with non-overlapping bands.
- [ ] **Follow-up on TTA (after current run locks):** flip `03_evaluation.ipynb` cell-14 to `get_d4_permutations(..., include_time_reversal=False)`; update cell-13 markdown (surviving group is spatial Klein-2 `{e, r2}`); rerun TTA; document delta vs. the 4-element baseline. Rationale: boundary-asymmetric noise in `rotated_memory_z` makes `T` only approximate.
- [ ] **Conditional next cycle (if d=5 recovers):** add soft `-log(p) · graph_dist` attention bias (Calibrated Physics Penalty). Math + plumbing already specified in prior plan-file drafts.
- [ ] **If d=5 still gaps with MWPM:** MWPM soft-label distillation in training loss (FIX 6).
- [ ] **Research Track 2:** QCT input embedding + graph-based hardware masking via JAX Pallas.
- [ ] **Research Track 3 (remaining):** in-model $D_4$ equivariant projections with weight sharing (beyond eval-side TTA already landed).
- [ ] **Research Track 4:** Mamba SSM temporal routing + Fourier-transform attention mods.
- [ ] **Research Track 5:** Profile TPU inference, evaluate pruning/quantization.

---
## Resume Prompt for Future Sessions
"Working on Transformer-based QEC decoder for rotated surface codes. Strict parameter efficiency focus (1.3M params). Roadmap in `PLAN.md` and `STATE.md`.
- **Collaborative Brainstorming:** Both Gemini and Claude models authorized for research/architectural brainstorming per documented roadmap.
- **Implementation Policy:** ONLY Claude-code authorized for code refactoring, structural implementation, direct file modification. Gemini models MUST NOT modify code — strictly limited to planning/research docs and collaborative analysis."

## Pause Prompt for Session End
"Before pausing:
1. Review all recent technical insights and research discussions.
2. Ensure `PLAN.md` and `STATE.md` reflect latest status, prioritized tracks, new constraints.
3. Verify all proposed changes documented, roadmap state clear for next session."
