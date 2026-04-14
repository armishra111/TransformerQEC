import math
from collections.abc import Sequence
from typing import Any


def _is_binary_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return isinstance(value, int | float) and math.isfinite(value) and value in {0, 1}


def logical_error_rate(predictions: Sequence[int | float | bool], labels: Sequence[int | float | bool]) -> float:
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not all(_is_binary_value(value) for value in predictions) or not all(
        _is_binary_value(value) for value in labels
    ):
        raise ValueError("predictions and labels must contain only finite binary values")

    mistakes = sum(int(pred != label) for pred, label in zip(predictions, labels, strict=True))
    return mistakes / len(labels)


def improvement_pct(mwpm_ler: float, transformer_ler: float) -> float:
    if not (
        math.isfinite(mwpm_ler)
        and math.isfinite(transformer_ler)
        and 0 <= mwpm_ler <= 1
        and 0 <= transformer_ler <= 1
    ):
        raise ValueError("LER values must be finite values in [0, 1]")
    if mwpm_ler == 0.0:
        return math.nan
    return 100.0 * (mwpm_ler - transformer_ler) / mwpm_ler
