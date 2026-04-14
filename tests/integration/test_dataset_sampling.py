from transformerqec.data.sampling import generate_dataset, sample_syndromes


def test_generate_dataset_shapes_are_consistent() -> None:
    batch = generate_dataset(distance=3, p_values=[0.005, 0.01], shots_per_p=32)
    assert batch.syndromes.shape == (64, 24)
    assert batch.labels.shape == (64,)
    assert batch.physical_error_rates.shape == (64,)
    assert batch.coords.shape == (24, 3)
