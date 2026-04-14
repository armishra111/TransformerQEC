from dataclasses import dataclass

import numpy as np

from transformerqec.codes.surface_code import extract_detector_coordinates, make_rotated_memory_z_circuit


@dataclass(frozen=True)
class DatasetBatch:
    syndromes: np.ndarray
    labels: np.ndarray
    physical_error_rates: np.ndarray
    coords: np.ndarray


def sample_syndromes(circuit, num_shots: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler()
    syndromes, observables = sampler.sample(num_shots, separate_observables=True)
    if observables.shape[1] == 0:
        raise ValueError("circuit must define at least one logical observable")
    return syndromes.astype(np.float32), observables[:, 0].astype(np.int64)


def generate_dataset(distance: int, p_values: list[float], shots_per_p: int) -> DatasetBatch:
    if not p_values:
        raise ValueError("p_values must not be empty")
    if shots_per_p <= 0:
        raise ValueError("shots_per_p must be positive")
    syndromes, labels, rates = [], [], []
    for p in p_values:
        sampled_syndromes, sampled_labels = sample_syndromes(
            make_rotated_memory_z_circuit(distance=distance, physical_error_rate=p),
            shots_per_p,
        )
        syndromes.append(sampled_syndromes)
        labels.append(sampled_labels)
        rates.append(np.full(shots_per_p, p, dtype=np.float32))
    return DatasetBatch(
        syndromes=np.concatenate(syndromes),
        labels=np.concatenate(labels),
        physical_error_rates=np.concatenate(rates),
        coords=extract_detector_coordinates(distance=distance),
    )
