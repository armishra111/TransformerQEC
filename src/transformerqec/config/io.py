import math
import numbers
from pathlib import Path
from typing import Any

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


SUPPORTED_SWEEP_SPACING = {"geomspace", "linspace"}


def _normalize_distance_key(field_name: str, raw_distance: object) -> int:
    if isinstance(raw_distance, str):
        try:
            distance = int(raw_distance)
        except ValueError as error:
            raise ValueError(f"{field_name} keys must be integers") from error
        _require(distance > 0, f"{field_name} keys must be positive integers")
        return distance
    _require(_is_positive_int(raw_distance), f"{field_name} keys must be positive integers")
    return int(raw_distance)


def _normalize_distance_map(field_name: str, raw_map: dict[object, object]) -> dict[int, int]:
    normalized = {}
    for raw_distance, raw_value in raw_map.items():
        _require(_is_positive_int(raw_value), f"{field_name} values must be positive integers")
        normalized[_normalize_distance_key(field_name, raw_distance)] = int(raw_value)
    return normalized


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_positive_distance_map(
    field_name: str,
    distance_map: dict[int, int],
    distances: list[int],
) -> None:
    missing_distances = sorted(set(distances) - set(distance_map))
    _require(not missing_distances, f"{field_name} must cover every data.distances value")

    invalid_values = [distance for distance, value in distance_map.items() if value <= 0]
    _require(not invalid_values, f"{field_name} values must be positive")


def _validate_sweep(field_name: str, sweep: SweepConfig) -> None:
    _require(
        sweep.spacing in SUPPORTED_SWEEP_SPACING,
        f"{field_name}.spacing must be one of {sorted(SUPPORTED_SWEEP_SPACING)}",
    )
    _require(_is_positive_int(sweep.count), f"{field_name}.count must be positive")
    _require(_is_finite_number(sweep.start), f"{field_name}.start must be finite")
    _require(_is_finite_number(sweep.stop), f"{field_name}.stop must be finite")
    if sweep.spacing == "geomspace":
        _require(sweep.start > 0, f"{field_name}.start must be positive for geomspace")
        _require(sweep.stop > 0, f"{field_name}.stop must be positive for geomspace")


def _validate_nonempty_path(field_name: str, value: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{field_name} must be non-empty")


def _validate_run_config(config: RunConfig) -> None:
    model = config.model
    data = config.data
    training = config.training
    evaluation = config.evaluation
    paths = config.paths

    _require(_is_positive_int(model.d_model), "model.d_model must be positive")
    _require(_is_positive_int(model.num_heads), "model.num_heads must be positive")
    _require(
        model.d_model % model.num_heads == 0,
        "model.d_model must be divisible by model.num_heads",
    )
    _require(model.pos_encoding == "rope", 'model.pos_encoding must be "rope"')
    _require(
        _is_finite_number(model.rope_spatial_ratio) and model.rope_spatial_ratio > 0,
        "model.rope_spatial_ratio must be positive",
    )
    _require(
        _is_finite_number(model.rope_temporal_ratio) and model.rope_temporal_ratio > 0,
        "model.rope_temporal_ratio must be positive",
    )

    _require(
        bool(data.distances) and all(_is_positive_int(distance) for distance in data.distances),
        "data.distances must be non-empty positive integers",
    )
    _validate_positive_distance_map(
        "model.num_layers_by_distance",
        model.num_layers_by_distance,
        data.distances,
    )
    _validate_positive_distance_map(
        "model.ffn_dim_by_distance",
        model.ffn_dim_by_distance,
        data.distances,
    )
    _require(data.noise_model == "phenomenological", 'data.noise_model must be "phenomenological"')
    _require(data.rounds_policy == "distance", 'data.rounds_policy must be "distance"')
    _validate_sweep("data.train_sweep", data.train_sweep)
    _validate_sweep("data.eval_sweep", data.eval_sweep)
    _require(
        _is_positive_int(data.total_train_samples),
        "data.total_train_samples must be positive",
    )
    _require(
        _is_finite_number(data.validation_fraction) and 0 < data.validation_fraction < 1,
        "data.validation_fraction must be between 0 and 1",
    )

    _require(_is_positive_int(training.batch_size), "training.batch_size must be positive")
    _require(_is_positive_int(training.num_epochs), "training.num_epochs must be positive")
    _require(
        _is_finite_number(training.peak_lr) and training.peak_lr > 0,
        "training.peak_lr must be finite and positive",
    )
    _require(
        _is_nonnegative_int(training.warmup_steps),
        "training.warmup_steps must be nonnegative",
    )
    _require(
        _is_finite_number(training.focal_gamma) and training.focal_gamma >= 0,
        "training.focal_gamma must be finite and nonnegative",
    )
    _require(
        _is_finite_number(training.focal_alpha) and 0 <= training.focal_alpha <= 1,
        "training.focal_alpha must be between 0 and 1",
    )
    _require(_is_nonnegative_int(training.seed), "training.seed must be a nonnegative integer")

    _require(_is_positive_int(evaluation.num_test), "evaluation.num_test must be positive")
    _validate_nonempty_path("evaluation.reference_csv", evaluation.reference_csv)
    _require(
        Path(evaluation.reference_csv).exists(),
        f"evaluation.reference_csv does not exist: {evaluation.reference_csv}",
    )
    configured_distances = set(data.distances)
    for pair in evaluation.threshold_pairs:
        _require(
            len(pair) == 2,
            "evaluation.threshold_pairs entries must have length 2",
        )
        _require(
            all(_is_positive_int(distance) for distance in pair),
            "evaluation.threshold_pairs entries must contain positive integers",
        )
        _require(
            all(distance in configured_distances for distance in pair),
            "evaluation.threshold_pairs entries must reference configured data.distances",
        )

    _validate_nonempty_path("paths.result_dir", paths.result_dir)
    _validate_nonempty_path("paths.baseline_dir", paths.baseline_dir)
    _validate_nonempty_path("paths.run_dir", paths.run_dir)
    _require(Path(paths.baseline_dir).exists(), f"paths.baseline_dir does not exist: {paths.baseline_dir}")


def materialize_sweep(sweep: SweepConfig) -> list[float]:
    if sweep.spacing == "geomspace":
        return np.geomspace(sweep.start, sweep.stop, sweep.count).tolist()
    if sweep.spacing == "linspace":
        return np.linspace(sweep.start, sweep.stop, sweep.count).tolist()
    raise ValueError(f"unsupported spacing: {sweep.spacing}")


def load_run_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    config = RunConfig(
        experiment_name=raw["experiment_name"],
        model=ModelConfig(
            d_model=raw["model"]["d_model"],
            num_heads=raw["model"]["num_heads"],
            num_layers_by_distance=_normalize_distance_map(
                "model.num_layers_by_distance",
                raw["model"]["num_layers_by_distance"],
            ),
            ffn_dim_by_distance=_normalize_distance_map(
                "model.ffn_dim_by_distance",
                raw["model"]["ffn_dim_by_distance"],
            ),
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
    _validate_run_config(config)
    return config
