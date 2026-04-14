import math
from collections.abc import Sequence


def logical_error_rate(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    mistakes = sum(int(pred != label) for pred, label in zip(predictions, labels, strict=True))
    return mistakes / len(labels)


def improvement_pct(mwpm_ler: float, transformer_ler: float) -> float:
    if mwpm_ler == 0.0:
        return math.nan
    return 100.0 * (mwpm_ler - transformer_ler) / mwpm_ler
