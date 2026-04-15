# Performance Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce avoidable setup and data-movement overhead in the decoder library without changing the current research baseline.

**Architecture:** Keep sampled syndromes compact until the model boundary, introduce cache helpers for deterministic setup work, and thread optional precomputed RoPE tables through the model/training seam so repeated runs can reuse them.

**Tech Stack:** Python 3.11, NumPy, JAX, Flax, Stim, PyMatching, pytest

---

### Task 1: Lock down the intended performance behavior with failing tests

**Files:**
- Modify: `tests/integration/test_dataset_sampling.py`
- Modify: `tests/integration/test_training_smoke.py`
- Modify: `tests/integration/test_benchmark_smoke.py`
- Modify: `tests/unit/test_rope.py`

- [ ] Add tests that require:
  - `sample_syndromes()` and `generate_dataset()` to keep syndrome arrays binary,
  - `train_step()` and/or `TransformerQEC` to accept bool syndromes and still produce finite outputs,
  - repeated sampler/decoder helper calls to reuse a cached compiled object,
  - repeated RoPE cache requests with the same coordinates/config to reuse cached tables.

- [ ] Run the focused tests and confirm they fail for the right reason before implementation.

### Task 2: Implement compact binary storage and reusable setup caches

**Files:**
- Modify: `src/transformerqec/data/sampling.py`
- Modify: `src/transformerqec/baselines/pymatching_decoder.py`
- Modify: `src/transformerqec/models/rope.py`
- Modify: `src/transformerqec/models/transformer.py`
- Modify: `src/transformerqec/training/loop.py`

- [ ] Add cached helper functions for compiled Stim samplers and PyMatching decoders.
- [ ] Keep syndrome arrays in bool form in sampling code.
- [ ] Add a RoPE cache helper plus optional precomputed RoPE inputs for the model/training seam.
- [ ] Cast bool/compact inputs once at the model boundary before dense layers.

### Task 3: Re-verify and document the resulting contract

**Files:**
- Modify: `docs/architecture.md` (only if needed to reflect the new reusable cache seam)

- [ ] Run the focused tests added above until green.
- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run ruff check src tests scripts`.
- [ ] Only update docs if the public architecture contract needs to mention the new cache helpers.
