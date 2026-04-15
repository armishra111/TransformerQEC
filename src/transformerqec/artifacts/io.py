import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CheckpointBundle:
    config: dict[str, Any]
    coords: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any | None]


def load_checkpoint_bundle(path: Path) -> CheckpointBundle:
    raw = pickle.loads(path.read_bytes())
    return CheckpointBundle(
        config=dict(raw["config"]),
        coords=np.asarray(raw["coords"], dtype=np.float32),
        params=dict(raw["params"]),
        metadata={
            "epoch": raw.get("epoch"),
            "val_loss": raw.get("val_loss"),
            "val_acc": raw.get("val_acc"),
        },
    )


def load_evaluation_rows(path: Path) -> list[dict[str, float]]:
    normalized_rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is not None:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for raw_row in reader:
            row = {key.strip(): value for key, value in raw_row.items()}
            normalized_rows.append(
                {
                    "distance": int(row["d"]),
                    "physical_error_rate": float(row["p"]),
                    "mwpm_ler": float(row["mwpm_ler"]),
                    "transformer_ler": float(row["transformer_ler"]),
                    "improvement_pct": float(row["improvement_pct"])
                    if row["improvement_pct"] != "nan"
                    else float("nan"),
                }
            )
    return normalized_rows
