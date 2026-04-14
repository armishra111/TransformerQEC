import numpy as np
import pymatching


def decode_with_pymatching(circuit, syndromes: np.ndarray) -> np.ndarray:
    syndrome_array = np.asarray(syndromes)
    if syndrome_array.ndim != 2:
        raise ValueError("syndromes must be a 2D array")
    if not np.issubdtype(syndrome_array.dtype, np.bool_) and not np.issubdtype(
        syndrome_array.dtype,
        np.number,
    ):
        raise ValueError("syndromes must contain only finite binary values")
    if not np.all(np.isfinite(syndrome_array)):
        raise ValueError("syndromes must contain only finite binary values")
    if not np.all((syndrome_array == 0) | (syndrome_array == 1)):
        raise ValueError("syndromes must contain only finite binary values")

    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(syndrome_array.astype(np.bool_, copy=False))
    if predictions.shape[1] == 0:
        raise ValueError("PyMatching returned no observable predictions")
    return np.asarray(predictions[:, 0], dtype=np.int64)
