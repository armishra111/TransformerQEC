"""Permutation-equivariance regression tests for TransformerQEC.

With DIPE (raw integer coords) and maskless attention, the model must be
invariant under any permutation of detectors that is also applied to the
`coords` array. Verified at random init — purely architectural property,
no training required.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model import TransformerQEC
from research_symmetries import (
    get_detector_coords,
    get_rot180_permutation,
    get_time_reversal_permutation,
)


def _build_model_and_inputs(d: int, seed: int = 0):
    coords = get_detector_coords(d).astype(np.float32)
    coords = coords - coords.min(axis=0, keepdims=True)   # DIPE origin-shift
    L = coords.shape[0]
    model = TransformerQEC(dtype=jnp.float32)             # f32 for tight tolerance
    key = jax.random.PRNGKey(seed)
    syn_key, init_key = jax.random.split(key)
    syn = jax.random.bernoulli(syn_key, 0.05, (1, L)).astype(jnp.float32)
    p = jnp.array([0.005], dtype=jnp.float32)
    params = model.init(init_key, syn, p, jnp.asarray(coords))["params"]
    return model, params, syn, p, coords


@pytest.mark.parametrize("d", [3, 5])
def test_rot180_invariance(d):
    model, params, syn, p, coords = _build_model_and_inputs(d)
    perm = get_rot180_permutation(coords)
    assert len(np.unique(perm)) == len(perm), "rot180 perm not bijective"
    out_a = model.apply({"params": params}, syn,          p, jnp.asarray(coords))
    out_b = model.apply({"params": params}, syn[:, perm], p, jnp.asarray(coords[perm]))
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-4)


@pytest.mark.parametrize("d", [3, 5])
def test_time_reversal_invariance(d):
    model, params, syn, p, coords = _build_model_and_inputs(d)
    perm = get_time_reversal_permutation(coords)
    assert len(np.unique(perm)) == len(perm), "time-reversal perm not bijective"
    out_a = model.apply({"params": params}, syn,          p, jnp.asarray(coords))
    out_b = model.apply({"params": params}, syn[:, perm], p, jnp.asarray(coords[perm]))
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-4)
