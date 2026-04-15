# Performance Slice 1 Design

**Goal:** Reduce avoidable Python/JAX overhead in the current library without changing baseline model behavior.

## Scope

This slice covers three low-risk improvements:

1. Keep sampled syndrome data in a compact binary representation until it reaches the model boundary.
2. Add reusable cache layers for expensive deterministic setup work:
   - Stim detector samplers used during dataset generation
   - PyMatching decoders built from a circuit detector error model
   - RoPE cosine/sine tables for a fixed coordinate layout and model head shape
3. Preserve the current public baseline behavior and checkpoint compatibility.

## Design

### Compact syndrome storage

`transformerqec.data.sampling` will stop eagerly converting sampled detector bits to `float32`. The stored batch representation will remain binary (`bool`) and the model boundary will perform the dtype conversion once before the first dense layer.

This keeps host memory, serialized artifacts, and host-to-device transfer smaller while preserving the current model math.

### Reusable deterministic caches

`transformerqec.data.sampling` and `transformerqec.baselines.pymatching_decoder` currently rebuild deterministic compiled helpers from the circuit on each call. This slice adds module-local LRU caches keyed by a canonical circuit text representation so repeated calls for the same circuit parameters reuse compiled setup.

`transformerqec.models.rope` will gain a cache helper keyed by the coordinate layout and RoPE hyperparameters. The model will accept optional precomputed RoPE tables so callers that already know the fixed detector layout can bypass table regeneration entirely.

### JAX boundary

The model remains the canonical coercion point for `syndrome`, `p_error`, and `coords`. This keeps callers simple and centralizes dtype behavior in the module that owns the numerical contract.

## Verification

The slice is complete when:

- dataset sampling tests show binary syndrome storage,
- training/model tests show bool syndromes still run correctly,
- cache-focused tests prove repeated calls reuse compiled helpers or precomputed tables,
- full `pytest` and `ruff` remain green.
