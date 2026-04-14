import csv
from pathlib import Path
from typing import Any


BENCHMARK_FIELDNAMES = ["d", "p", "mwpm_ler", "transformer_ler", "improvement_pct"]


def write_benchmark_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        missing_fields = [field for field in BENCHMARK_FIELDNAMES if field not in row]
        if missing_fields:
            fields = ", ".join(missing_fields)
            raise ValueError(f"benchmark row {index} is missing required fields: {fields}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_threshold_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
