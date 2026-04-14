import numpy as np

from transformerqec.codes.surface_code import (
    extract_detector_coordinates,
    make_rotated_memory_z_circuit,
)


def test_d3_detector_count_matches_current_checkpoint() -> None:
    circuit = make_rotated_memory_z_circuit(distance=3, physical_error_rate=0.01)
    assert circuit.num_detectors == 24


def test_detector_coordinates_are_normalized() -> None:
    coords = extract_detector_coordinates(distance=5)
    assert coords.shape == (120, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords <= 1.0)
