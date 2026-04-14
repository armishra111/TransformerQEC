import csv
from pathlib import Path
from typing import Any


BENCHMARK_FIELDNAMES = ["d", "p", "mwpm_ler", "transformer_ler", "improvement_pct"]


def write_benchmark_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_threshold_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
