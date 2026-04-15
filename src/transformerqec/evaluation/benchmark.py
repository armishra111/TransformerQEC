import csv
import math
import numbers
from pathlib import Path
from typing import Any


BENCHMARK_FIELDNAMES = ["d", "p", "mwpm_ler", "transformer_ler", "improvement_pct"]


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value > 0


def _is_probability(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _is_finite_or_nan_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and not math.isinf(value)


def _validate_benchmark_row(index: int, row: dict[str, Any]) -> None:
    missing_fields = [field for field in BENCHMARK_FIELDNAMES if row.get(field) is None]
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ValueError(f"benchmark row {index} is missing required fields: {fields}")

    if not _is_positive_integer(row["d"]):
        raise ValueError(f"benchmark row {index} has invalid d")
    for field in ("p", "mwpm_ler", "transformer_ler"):
        if not _is_probability(row[field]):
            raise ValueError(f"benchmark row {index} has invalid {field}")
    if not _is_finite_or_nan_real(row["improvement_pct"]):
        raise ValueError(f"benchmark row {index} has invalid improvement_pct")


def write_benchmark_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        _validate_benchmark_row(index, row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_threshold_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
