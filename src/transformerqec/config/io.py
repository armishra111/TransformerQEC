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


def materialize_sweep(sweep: SweepConfig) -> list[float]:
    if sweep.spacing == "geomspace":
        return np.geomspace(sweep.start, sweep.stop, sweep.count).tolist()
    return np.linspace(sweep.start, sweep.stop, sweep.count).tolist()


def load_run_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    return RunConfig(
        experiment_name=raw["experiment_name"],
        model=ModelConfig(**raw["model"]),
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
