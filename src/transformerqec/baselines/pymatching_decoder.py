from functools import lru_cache

import numpy as np
import pymatching
import stim


@lru_cache(maxsize=128)
def _build_matching(circuit_text: str) -> pymatching.Matching:
    circuit = stim.Circuit(circuit_text)
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def _get_matching(circuit) -> pymatching.Matching:
    return _build_matching(str(circuit))


def decode_with_pymatching(circuit, syndromes: np.ndarray) -> np.ndarray:
    """Return the first observable prediction from the PyMatching baseline."""
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

    matching = _get_matching(circuit)
    predictions = matching.decode_batch(syndrome_array.astype(np.bool_, copy=False))
    if predictions.shape[1] == 0:
        raise ValueError("PyMatching returned no observable predictions")
    return np.asarray(predictions[:, 0], dtype=np.int64)
