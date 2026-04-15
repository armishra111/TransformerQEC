from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim

from transformerqec.codes.surface_code import extract_detector_coordinates, make_rotated_memory_z_circuit


@dataclass(frozen=True)
class DatasetBatch:
    syndromes: np.ndarray
    labels: np.ndarray
    physical_error_rates: np.ndarray
    coords: np.ndarray


@lru_cache(maxsize=128)
def _compile_detector_sampler(circuit_text: str):
    return stim.Circuit(circuit_text).compile_detector_sampler()


def _get_compiled_detector_sampler(circuit) -> object:
    return _compile_detector_sampler(str(circuit))


def sample_syndromes(circuit, num_shots: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = _get_compiled_detector_sampler(circuit)
    syndromes, observables = sampler.sample(num_shots, separate_observables=True)
    if observables.shape[1] == 0:
        raise ValueError("circuit must define at least one logical observable")
    return syndromes.astype(np.bool_, copy=False), observables[:, 0].astype(np.int64, copy=False)


def generate_dataset(distance: int, p_values: list[float], shots_per_p: int) -> DatasetBatch:
    if not p_values:
        raise ValueError("p_values must not be empty")
    if shots_per_p <= 0:
        raise ValueError("shots_per_p must be positive")

    total_shots = len(p_values) * shots_per_p
    syndrome_buffer: np.ndarray | None = None
    label_buffer = np.empty(total_shots, dtype=np.int64)
    rate_buffer = np.empty(total_shots, dtype=np.float32)

    for index, p in enumerate(p_values):
        sampled_syndromes, sampled_labels = sample_syndromes(
            make_rotated_memory_z_circuit(distance=distance, physical_error_rate=p),
            shots_per_p,
        )
        if syndrome_buffer is None:
            syndrome_buffer = np.empty((total_shots, sampled_syndromes.shape[1]), dtype=np.bool_)

        start = index * shots_per_p
        stop = start + shots_per_p
        syndrome_buffer[start:stop] = sampled_syndromes
        label_buffer[start:stop] = sampled_labels
        rate_buffer[start:stop] = p

    assert syndrome_buffer is not None
    return DatasetBatch(
        syndromes=syndrome_buffer,
        labels=label_buffer,
        physical_error_rates=rate_buffer,
        coords=extract_detector_coordinates(distance=distance),
    )
