import numpy as np
import stim


def make_rotated_memory_z_circuit(
    distance: int,
    physical_error_rate: float,
    rounds: int | None = None,
    noise_model: str = "phenomenological",
) -> stim.Circuit:
    if rounds is None:
        rounds = distance
    kwargs = {"distance": distance, "rounds": rounds}
    if noise_model == "code_capacity":
        kwargs.update({"rounds": 1, "before_round_data_depolarization": physical_error_rate})
    elif noise_model == "phenomenological":
        kwargs.update(
            {
                "before_round_data_depolarization": physical_error_rate,
                "before_measure_flip_probability": physical_error_rate,
            }
        )
    else:
        raise ValueError(f"unsupported noise_model: {noise_model!r}")
    return stim.Circuit.generated("surface_code:rotated_memory_z", **kwargs)


def extract_detector_coordinates(distance: int, rounds: int | None = None) -> np.ndarray:
    circuit = make_rotated_memory_z_circuit(distance=distance, physical_error_rate=0.01, rounds=rounds)
    raw = circuit.get_detector_coordinates()
    coords = np.zeros((circuit.num_detectors, 3), dtype=np.float32)
    for detector_index, coord_tuple in raw.items():
        coords[detector_index] = coord_tuple[:3]
    for axis in range(3):
        low, high = coords[:, axis].min(), coords[:, axis].max()
        if high > low:
            coords[:, axis] = (coords[:, axis] - low) / (high - low)
    return coords
