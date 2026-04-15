import numpy as np
import pytest

import transformerqec.data.sampling as sampling
from transformerqec.data.sampling import generate_dataset, sample_syndromes


def test_generate_dataset_shapes_are_consistent() -> None:
    batch = generate_dataset(distance=3, p_values=[0.005, 0.01], shots_per_p=32)
    assert batch.syndromes.shape == (64, 24)
    assert batch.labels.shape == (64,)
    assert batch.physical_error_rates.shape == (64,)
    assert batch.coords.shape == (24, 3)
    assert batch.syndromes.dtype == np.bool_


def test_sample_syndromes_returns_shapes_and_dtypes() -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    syndromes, labels = sample_syndromes(circuit, num_shots=5)

    assert syndromes.shape == (5, 1)
    assert labels.shape == (5,)
    assert syndromes.dtype == np.bool_
    assert labels.dtype == np.int64


def test_compiled_detector_sampler_is_reused_for_equivalent_circuits() -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    equivalent_circuit = stim.Circuit(str(circuit))

    first = sampling._get_compiled_detector_sampler(circuit)
    second = sampling._get_compiled_detector_sampler(equivalent_circuit)

    assert first is second


def test_sample_syndromes_requires_logical_observable() -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]")

    with pytest.raises(ValueError, match="circuit must define at least one logical observable"):
        sample_syndromes(circuit, num_shots=5)


def test_generate_dataset_rejects_empty_p_values() -> None:
    with pytest.raises(ValueError, match="p_values must not be empty"):
        generate_dataset(distance=3, p_values=[], shots_per_p=32)


def test_generate_dataset_rejects_non_positive_shots() -> None:
    with pytest.raises(ValueError, match="shots_per_p must be positive"):
        generate_dataset(distance=3, p_values=[0.01], shots_per_p=0)
