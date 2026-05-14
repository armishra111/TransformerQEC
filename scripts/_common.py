"""Shared helpers for the baseline reproduction scripts.

Matches commit cc3abc5 (the architecture that produced results/legacy/*.pkl
and results/evaluation_results.csv). Coords are normalized to [0, 1] per
axis — the RoPE module re-scales them by seq_len. Cross-entropy loss.
Bulk-materialized dataset (not on-the-fly).
"""
from __future__ import annotations

import numpy as np
import stim


def make_circuit(d: int, p: float, rounds: int | None = None) -> stim.Circuit:
    if rounds is None:
        rounds = d
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )


def sample_syndromes(circuit: stim.Circuit, num_shots: int):
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(num_shots, separate_observables=True)
    return det.astype(np.float32), obs[:, 0].astype(np.int64)


def get_detector_coords(d: int, rounds: int | None = None) -> np.ndarray:
    """Normalized detector (x, y, t) coords in [0, 1] per axis (legacy form)."""
    circuit = make_circuit(d, p=0.01, rounds=rounds)
    raw = circuit.get_detector_coordinates()
    coords = np.zeros((circuit.num_detectors, 3), dtype=np.float32)
    for det_idx, c in raw.items():
        c3 = np.asarray(c[:3], dtype=np.float32)
        coords[det_idx, : c3.shape[0]] = c3
    for axis in range(3):
        lo, hi = coords[:, axis].min(), coords[:, axis].max()
        if hi > lo:
            coords[:, axis] = (coords[:, axis] - lo) / (hi - lo)
    return coords


def generate_dataset(d: int, p_values, shots_per_p: int):
    """Bulk-materialize a (syndrome, label, p, coords) dataset.

    `coords` returned with shape (seq_len, 3) — same coords across all p.
    """
    all_syn, all_lab, all_p = [], [], []
    for p in p_values:
        syn, lab = sample_syndromes(make_circuit(d, p), shots_per_p)
        all_syn.append(syn)
        all_lab.append(lab)
        all_p.append(np.full(shots_per_p, p, dtype=np.float32))
    coords = get_detector_coords(d)
    return (
        np.concatenate(all_syn),
        np.concatenate(all_lab),
        np.concatenate(all_p),
        coords,
    )


def wilson_ci(successes: int, total: int, z: float = 1.96):
    """Wilson 95% CI for a Bernoulli proportion."""
    if total == 0:
        return 0.0, 1.0
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    spread = z * np.sqrt(
        p_hat * (1.0 - p_hat) / total + z * z / (4.0 * total * total)
    ) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def cross_entropy_loss(logits, labels, num_classes: int = 2):
    import jax
    import jax.numpy as jnp
    one_hot = jax.nn.one_hot(labels, num_classes)
    log_probs = jax.nn.log_softmax(logits)
    return -(one_hot * log_probs).sum(-1).mean()
