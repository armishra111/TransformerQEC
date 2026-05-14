# Project Plan
# Last Updated: 2026-04-17

## Objectives
- [x] Train 1.3M param Transformer encoder for QEC on rotated surface codes.
- [x] Implement physics-informed (2+1)D Anisotropic RoPE for lattice structures.
- [x] Compare Transformer vs MWPM baseline for $d \in \{3, 5, 7\}$.
- [x] Validate scaling behavior ($p_L \propto p^{(d+1)/2}$) and estimate thresholds.
- [ ] Refactor codebase to `src/` modular structure (restricted).
- [ ] Implement symmetry-aware and distance-agnostic decoding optimizations (restricted).

## Status & Constraints
- **Constraint:** Strictly restricted from code refactoring or implementation. Role limited to updating `plan.md` and `STATE.md` for roadmap.
- **Model Efficiency Goal:** Core constraint = parameter efficiency (maintain 1.3M param success). Future architectures must prioritize elegant inductive biases over brute-force data scaling.

## Recently Implemented (Cycle A: 2026-04-17 — architecture fixes)
- [x] **Train/eval architecture unification.** `notebooks/02_model_and_training.ipynb` now `import`s `TransformerQEC` from `notebooks/model.py` instead of carrying an inline copy. Single source of truth; train/eval cannot drift again.
- [x] **Locality mask removed.** `create_locality_mask` deleted from `notebooks/model.py`; mask plumbing removed from `TransformerBlockWithRoPE`. Restores maskless attention that historically beat MWPM at d=5.
- [x] **DIPE landed (Phase 3 Priority 1 - CRITICAL).** `build_rope_2_5d` now consumes raw integer detector coordinates (origin-shifted only, no per-distance rescaling). RoPE base frequencies retuned for the small surface-code positional ranges (`base_spatial=100`, `base_temporal=20`). Same physical lattice cell yields the same RoPE angle at any `d`. `code_distance`/`measurement_rounds` fields kept on `TransformerQEC` for checkpoint backcompat but unused at runtime.
- **Backcompat note:** existing `transformer_qec_d{3,5,7}.pkl` checkpoints are no longer loadable (Q/K projections trained against the old RoPE geometry). New checkpoints written under tag `model_version: 'dipe-no-mask'`.

## Recently Implemented (Cycle B: 2026-04-17 — eval-side scaffolding)
- [x] **Wilson 95% CI bands on LER plots.** `notebooks/03_evaluation.ipynb` cell-12 computes Wilson CI per (d, p) for both MWPM and Transformer; per-decoder (cell-15) and combined (cell-16) plots render `fill_between` bands. CSV (cell-18) carries four new CI columns. Previously raw LER points carried no uncertainty — meaningless at p=0.001 where expected positives ~10.
- [x] **Checkpoint version guard.** cell-10 requires `ckpt['config']['model_version'] == 'dipe-no-mask'`; legacy pkls skip with explicit warning instead of silently producing nonsense LERs. Checkpoint file pattern renamed to `best_d{d}_checkpoint.pkl` (matches training output).
- [x] **FIX 3: Equivariance regression tests.** `tests/` un-gitignored; new `tests/test_equivariance.py` parametrized over d ∈ {3, 5}; `rot180` and `time-reversal` permutation invariance verified at random init in float32. **All 4 tests pass locally in 13.7s** — confirms DIPE math + maskless attention preserve the architectural invariants claimed. `research_symmetries.py` bottom sweep wrapped in `if __name__ == "__main__":` so imports are silent.
- [x] **FIX 8: D₄ × time-reversal test-time augmentation.** New `get_d4_permutations(coords)` in `research_symmetries.py` enumerates 8 spatial × 2 time = 16 candidates, filters to valid bijective detector permutations. For `rotated_memory_z` only `{e, r2, eT, r2T}` survive (90°/270°/diagonals swap X/Z stabilizer types and are correctly rejected). New cell-14 in `03_evaluation.ipynb` averages softmax probabilities over the surviving group elements via `jax.vmap`. Expected ~1-3% LER reduction; plots overlay `Transformer+TTA d=X` curves with their own CI bands.
    - **⚠ Known caveat (documented 2026-04-18, code change deferred):** the current 4-element group includes `eT` / `r2T`. Time-reversal `T` is only an *approximate* symmetry of `rotated_memory_z` because the circuit is asymmetric at its boundaries — noiseless `|0⟩` init vs. noisy final data-qubit measurement (`before_measure_flip_probability=p`). The `t=0` and `t=rounds` boundary detectors see different noise budgets, so averaging over `*T` elements biases the softmax mean by a non-symmetry term. Planned fix: pass `include_time_reversal=False` in cell-14 → Klein-2 `{e, r2}`, which is an exact syndrome-distribution symmetry. **Deferred until the current 4-element eval run locks** so 4-element vs. 2-element TTA LERs can be compared side-by-side against the same checkpoint.

## Recently Implemented (Cycle C: 2026-04-17 — pre-run hygiene + nn.remat)
- [x] **FIX 10: Gradient checkpointing (`nn.remat`).** `notebooks/model.py:302-312`: `TransformerBlockWithRoPE` wrapped in `nn.remat` so block activations are recomputed in backward pass instead of stored. ~$\mathcal{O}(1/\sqrt{N_{\text{layers}}})$ activation memory at ~30% compute cost. Forward output bit-identical — existing `dipe-no-mask` checkpoints load unchanged. Unlocks d=9 training headroom. Equivariance tests still pass (4/4 in 12.15s post-remat).
- [x] **Timestamped checkpoint filenames.** `02_model_and_training.ipynb` cell-13 generates `RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')`; cell-14 writes `best_d{d}_{RUN_STAMP}_checkpoint.pkl` and `safety_d{d}_{RUN_STAMP}_epoch{N}.pkl`. Every checkpoint config also carries a `'timestamp': RUN_STAMP` field. `03_evaluation.ipynb` cell-9 parses the timestamped pattern via regex, groups by distance; cell-10 picks latest per `d` by lexicographic ISO sort, with optional `CHECKPOINT_OVERRIDES` dict for pinning specific stamps. Repeated training runs no longer overwrite each other.
- [x] **Legacy checkpoint archive.** `results/transformer_qec_d{3,5,7}.pkl` moved to `results/legacy/` (git renames staged, uncommitted). New `best_d*_checkpoint.pkl` pattern keeps `results/` clean.
- [x] **Minor hygiene:** `research_symmetries.py:9` defensive slice (`coords[idx, :c3.shape[0]] = c3`) for variable-length Stim coords; duplicate `drive.mount()` removed from training cell-13; eval cell-0 markdown updated to reference the new timestamped filename pattern; training cell-5 markdown architecture description aligned with post-DIPE + post-remat reality.

## Recently Implemented (Cycle D: 2026-04-17 — on-the-fly STIM pipeline)
- [x] **`requirements.txt` at repo root** pinning `stim>=1.14,<2`, `pymatching>=2`, `flax>=0.8`, `jax[tpu]>=0.4`, `optax>=0.2`, `numpy<2`. Both notebooks' cell-1 install via `!pip install -q -r /content/drive/MyDrive/TransformerQEC/requirements.txt` instead of free-floating version constraints.
- [x] **On-the-fly STIM sampling.** Replaces the 6M-sample bulk-materialized dataset with chunk-level generation during training. Each of the `STEPS_PER_EPOCH × NUM_EPOCHS` training steps now sees a distinct STIM sampler draw — effective dataset size unbounded, host RAM footprint constant, zero memorization risk.
  - `02_model_and_training.ipynb` cell-4: `build_samplers(d, p_values)` caches `stim.Circuit.compile_detector_sampler()` per `p` value; `generate_val_dataset` preserves the bulk pattern for the validation set (stability, deterministic seed).
  - cell-8: `train_chunk` = `jax.lax.scan` over `CHUNK_SIZE=100` pre-sampled batches per TPU call → retains scan throughput. `eval_epoch` unchanged.
  - cell-14: `ThreadPoolExecutor` prefetches chunk N+1 on CPU while TPU trains chunk N (~200ms sample vs ~3s train at d=5 → CPU hidden). Non-uniform p-schedule (weights ∝ $1/\sqrt{p}$) oversamples low-p bins, subsuming FIX 9 stratified sampling. Step-rate monitor logs steps/sec every 100 steps.
- [x] **Checkpoint version tag bumped to `'dipe-no-mask-otf'`**. `03_evaluation.ipynb` cell-10 `EXPECTED_MODEL_VERSIONS = {'dipe-no-mask', 'dipe-no-mask-otf'}` accepts both bulk and OTF checkpoints without further eval edits.

## Milestones (Pending Approval for Implementation)
### Phase 1: Modularization & On-the-fly Data Generation
- [ ] Init `src/` package structure and `setup.py` for notebook imports.
- [ ] Centralize `TransformerQEC` and RoPE logic into `src/architecture/`.
- [ ] Abstract STIM wrappers into `src/data/`.
- [ ] Consolidate JAX utilities into `src/training/`.
- [x] Implement on-the-fly STIM pipeline. **(Landed 2026-04-17 Cycle D — chunk-scan + prefetch + non-uniform p schedule.)**

### Phase 2: Training & Evaluation (Notebook-Driven)
- [ ] Refactor notebooks as driver scripts for TPU training.
- [ ] Train/benchmark models.

### Phase 3: Advanced Optimization & Hardware Alignment (Deep Technical Focus)
- [x] **Distance-Invariant Position Encoding (DIPE) & Distance Scaling (Priority 1 - CRITICAL):**
    - [x] Implement Lattice-Locked RoPE Frequencies. Anchor rotation matrix to absolute basis frequencies tied to exact integer lattice coordinates $(dx, dy)$ — eliminate angular drift. **(Landed 2026-04-17 in `model.py:build_rope_2_5d`.)**
    - [ ] Implement unified model training across variable $d$ using relative symmetry invariants. **(Deferred — gated on the d=5 result of the DIPE-only run.)**
- [ ] **Sequence Reduction & Hardware-Aware Masking (Priority 2 - HIGH):**
    - **Qubit-Centric Projection (QCT):** Deprecate naive binary syndrome embedding. Concatenate adjacent $X$ and $Z$ stabilizer events into unified physical qubit features — halve spatial sequence length, mitigate $\mathcal{O}(d^4)$ bottlenecks.
    - **Graph-Based Hardware Masking:** Replace arbitrary Manhattan radius heuristics with exact physical adjacency matrix. Deploy JAX Pallas kernels for TPU v6e MXUs, memory efficiency. **(Hard Manhattan mask removed 2026-04-17; graph-adjacency replacement still pending.)**
    - Explore **Calibrated Physics Penalty**: $AttentionScore - \beta \cdot [-\log(p)] \cdot GraphDistance$ — dynamically scale receptive fields. **(Queued for next cycle, conditional on DIPE-only d=5 recovering to maskless-baseline territory.)**
- [~] **Symmetry-Aware Decoding (Priority 3 - Physics-Inspired):**
    - [x] Implement Test-Time Augmentation (TTA) via 3D spacetime point inversion ($pos \to -pos$, $time \to -time$). **(Landed 2026-04-17 Cycle B — `get_d4_permutations` + vmap'd softmax-average in `03_evaluation.ipynb` cell-14. For rotated_memory_z only the Z₂×Z₂ subgroup {e, r2, eT, r2T} is physical — 90° rotations swap X/Z stabilizer types and are auto-rejected.)**
        - [ ] **Follow-up (queued 2026-04-18):** after current 4-element TTA eval locks, flip cell-14 to `include_time_reversal=False` → Klein-2 `{e, r2}` only. `T` is only approximate for `rotated_memory_z` (boundary-asymmetric noise: noiseless init vs. noisy final measurement), so `*T` elements currently bias softmax mean. Rerun and compare against the 4-element baseline.
    - [x] Maintain distinct spatial and temporal RoPE/locality mechanisms. **(Already satisfied: 2.5D RoPE splits head dims 3:1 spatial:temporal.)**
    - [ ] Implement $D_4$ group-equivariant layers using group-invariant projection bases and symmetry-weight sharing in attention. **(TTA is runtime averaging; in-model equivariance with weight sharing still pending.)**
    - [ ] **Hybrid RoPE + Reflection-Symmetric Additive Bias `B_sym(|\Delta x|, |\Delta y|, |\Delta t|)` (queued, design locked 2026-04-18):** Add small learnable additive bias to attention logits pre-softmax, on top of existing 2.5D RoPE. Bias is content-blind, even in each axis → enforces $\mathbb{Z}_2 \times \mathbb{Z}_2$ reflection-symmetric inductive prior + soft locality. RoPE kept intact; sin DOF retained so model can still learn anisotropy/hook-tilt where data demands. Cheap (~hundreds of floats per head, ~0 compute overhead) — aligned with 1.3M param-efficiency constraint. **Noise-model applicability:**
        - **Phenomenological (current synthetic):** $D_4$ approximately exact in bulk (boundary-asymmetric for `rotated_memory_z`). Bias acts as near-exact prior in bulk, soft prior at boundary. Strong regularizer, near-zero risk.
        - **Circuit-level:** $D_4$ broken by CNOT scheduling order (hook-error tilt) and X/Z stabilizer asymmetry. Bias acts as *inductive prior* toward symmetric solution; RoPE sin DOF lets model learn the residual asymmetry. Hybrid (not pure even kernel) is the *correct* baseline — forcing hard symmetry would erase real predictive signal from hook tilt.
        - **Real hardware (Sycamore/Willow):** $D_4$ further broken by per-site $T_1, T_2$, gate fidelity, leakage, readout asymmetry. Bias still useful as locality prior but must be paired with per-site calibration features (separate work item). Do not enforce hard symmetry.
        - See `DEEP_RESEARCH_FIX_SUGGS.md` FIX 12 for implementation plan.
- [ ] **Temporal Processing & Noise Decoupling (Priority 4 - Exploratory):**
    - Investigate **Mamba-Augmented Temporal Routing** (Selective State-Space Models) replacing temporal self-attention — target linear $\mathcal{O}(d^2)$ temporal complexity.
    - Research **Spectral/Fourier Attention Gating** (FFT-based attention) or **Dual-Stream** mechanism — decouple high-freq topological errors from low-freq hardware leakage.
- [~] **Hardware Alignment & Latency Optimization (Priority 5 - TPU Target):**
    - [x] Enforce precision discipline (Q-K matmuls in `bfloat16`, upcast `float32` for `softmax`, cast back `bfloat16`). **(Already in `model.py`.)**
    - [x] Gradient checkpointing via `nn.remat`. **(Landed 2026-04-17 Cycle C.)**
    - [ ] Static compilation: Pad all windows to constant size — prevent XLA recompilation.
    - [ ] VMEM Management: Pallas-orchestrated DMA transfers HBM→VMEM.
    - [ ] Profile TPU inference throughput, evaluate pruning/quantization.

---
## Resume Prompt for Future Sessions
"Working on Transformer-based QEC decoder for rotated surface codes, strict parameter efficiency focus (1.3M params). Roadmap in `PLAN.md` and `STATE.md`.
- **Collaborative Brainstorming:** Both Gemini and Claude authorized for research/architectural brainstorming per documented roadmap.
- **Implementation Policy:** ONLY Claude-code authorized for code refactoring, structural implementation, direct file modification. Gemini models MUST NOT modify code — strictly limited to planning/research docs and collaborative analysis."

## Pause Prompt for Session End
"Before pausing:
1. Review all recent technical insights and research discussions.
2. Ensure `PLAN.md` and `STATE.md` reflect latest project status, prioritized tracks, new constraints.
3. Verify all proposed changes documented, roadmap state clear for next session."