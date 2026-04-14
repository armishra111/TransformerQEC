from pathlib import Path

import numpy as np
import yaml

from transformerqec.config.schema import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
    RunConfig,
    SweepConfig,
    TrainingConfig,
)


def _normalize_distance_map(raw_map: dict[object, object]) -> dict[int, int]:
    return {int(distance): int(value) for distance, value in raw_map.items()}


def materialize_sweep(sweep: SweepConfig) -> list[float]:
    if sweep.spacing == "geomspace":
        return np.geomspace(sweep.start, sweep.stop, sweep.count).tolist()
    if sweep.spacing == "linspace":
        return np.linspace(sweep.start, sweep.stop, sweep.count).tolist()
    raise ValueError(f"unsupported spacing: {sweep.spacing}")


def load_run_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    return RunConfig(
        experiment_name=raw["experiment_name"],
        model=ModelConfig(
            d_model=raw["model"]["d_model"],
            num_heads=raw["model"]["num_heads"],
            num_layers_by_distance=_normalize_distance_map(raw["model"]["num_layers_by_distance"]),
            ffn_dim_by_distance=_normalize_distance_map(raw["model"]["ffn_dim_by_distance"]),
            pos_encoding=raw["model"]["pos_encoding"],
            rope_spatial_ratio=raw["model"]["rope_spatial_ratio"],
            rope_temporal_ratio=raw["model"]["rope_temporal_ratio"],
        ),
        data=DataConfig(
            distances=raw["data"]["distances"],
            noise_model=raw["data"]["noise_model"],
            rounds_policy=raw["data"]["rounds_policy"],
            train_sweep=SweepConfig(**raw["data"]["train_sweep"]),
            eval_sweep=SweepConfig(**raw["data"]["eval_sweep"]),
            total_train_samples=raw["data"]["total_train_samples"],
            validation_fraction=raw["data"]["validation_fraction"],
        ),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        paths=PathsConfig(**raw["paths"]),
    )
