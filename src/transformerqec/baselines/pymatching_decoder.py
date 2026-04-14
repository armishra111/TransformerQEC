import numpy as np
import pymatching


def decode_with_pymatching(circuit, syndromes: np.ndarray) -> np.ndarray:
    syndrome_array = np.asarray(syndromes)
    if syndrome_array.ndim != 2:
        raise ValueError("syndromes must be a 2D array")

    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model())
    predictions = matching.decode_batch(syndrome_array.astype(np.bool_, copy=False))
    if predictions.shape[1] == 0:
        raise ValueError("PyMatching returned no observable predictions")
    return np.asarray(predictions[:, 0], dtype=np.int64)
