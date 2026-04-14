from pathlib import Path
from typing import Any

import pytest
import yaml

from transformerqec.config.io import load_run_config, materialize_sweep


def write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def valid_config(tmp_path: Path) -> dict[str, Any]:
    reference_csv = tmp_path / "evaluation_results.csv"
    reference_csv.write_text("d,p,mwpm_ler,transformer_ler,improvement_pct\n")
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    return {
        "experiment_name": "valid",
        "model": {
            "d_model": 128,
            "num_heads": 4,
            "num_layers_by_distance": {3: 4, 5: 6},
            "ffn_dim_by_distance": {3: 1024, 5: 512},
            "pos_encoding": "rope",
            "rope_spatial_ratio": 3,
            "rope_temporal_ratio": 1,
        },
        "data": {
            "distances": [3, 5],
            "noise_model": "phenomenological",
            "rounds_policy": "distance",
            "train_sweep": {
                "start": 0.005,
                "stop": 0.01,
                "count": 2,
                "spacing": "geomspace",
            },
            "eval_sweep": {
                "start": 0.005,
                "stop": 0.01,
                "count": 2,
                "spacing": "geomspace",
            },
            "total_train_samples": 4096,
            "validation_fraction": 0.125,
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 1,
            "peak_lr": 0.0001,
            "warmup_steps": 8,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "seed": 7,
        },
        "evaluation": {
            "num_test": 2048,
            "reference_csv": str(reference_csv),
            "threshold_pairs": [[3, 5]],
        },
        "paths": {
            "result_dir": str(tmp_path / "results"),
            "baseline_dir": str(baseline_dir),
            "run_dir": str(tmp_path / "runs"),
        },
    }


def test_current_config_materializes_expected_training_grid() -> None:
    config = load_run_config(Path("configs/baseline/current.yaml"))
    train_grid = materialize_sweep(config.data.train_sweep)
    assert len(train_grid) == 20
    assert round(train_grid[0], 6) == 0.002
    assert round(train_grid[-1], 6) == 0.017
    assert config.evaluation.reference_csv == "results/baseline/evaluation_results.csv"


def test_smoke_config_stays_small() -> None:
    config = load_run_config(Path("configs/laptop/d3-smoke.yaml"))
    assert config.data.distances == [3]
    assert config.training.batch_size == 64
    assert config.evaluation.num_test == 2048
    assert config.evaluation.reference_csv == "results/baseline/evaluation_results.csv"


def test_materialize_sweep_rejects_unknown_spacing() -> None:
    with pytest.raises(ValueError, match="unsupported spacing"):
        materialize_sweep(
            type("Sweep", (), {"start": 0.1, "stop": 0.2, "count": 3, "spacing": "badspace"})()
        )


def test_quoted_distance_keys_load_as_int_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "quoted-keys.yaml"
    config_path.write_text(
        """
experiment_name: quoted-keys
model:
  d_model: 128
  num_heads: 4
  num_layers_by_distance: {"3": 4}
  ffn_dim_by_distance: {"3": 1024}
  pos_encoding: rope
  rope_spatial_ratio: 3
  rope_temporal_ratio: 1
data:
  distances: [3]
  noise_model: phenomenological
  rounds_policy: distance
  train_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  eval_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  total_train_samples: 4096
  validation_fraction: 0.125
training:
  batch_size: 64
  num_epochs: 1
  peak_lr: 0.0001
  warmup_steps: 8
  focal_gamma: 2.0
  focal_alpha: 0.75
  seed: 7
evaluation:
  num_test: 2048
  reference_csv: results/baseline/evaluation_results.csv
  threshold_pairs: [[3, 3]]
paths:
  result_dir: results
  baseline_dir: results/baseline
  run_dir: results/runs
""".strip()
    )

    config = load_run_config(config_path)
    assert config.model.num_layers_by_distance[3] == 4
    assert config.model.ffn_dim_by_distance[3] == 1024


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (lambda cfg: cfg["model"].update(d_model=0), "model.d_model"),
        (lambda cfg: cfg["model"].update(num_heads=0), "model.num_heads"),
        (lambda cfg: cfg["model"].update(d_model=130), "model.d_model"),
        (lambda cfg: cfg["model"].update(pos_encoding="absolute"), "model.pos_encoding"),
        (lambda cfg: cfg["model"].update(rope_spatial_ratio=0), "model.rope_spatial_ratio"),
        (lambda cfg: cfg["model"].update(rope_temporal_ratio=0), "model.rope_temporal_ratio"),
        (lambda cfg: cfg["data"].update(distances=[]), "data.distances"),
        (lambda cfg: cfg["data"].update(distances=[3, 0]), "data.distances"),
        (
            lambda cfg: cfg["model"]["num_layers_by_distance"].pop(5),
            "model.num_layers_by_distance",
        ),
        (
            lambda cfg: cfg["model"]["ffn_dim_by_distance"].update({5: 0}),
            "model.ffn_dim_by_distance",
        ),
        (lambda cfg: cfg["data"].update(noise_model="circuit"), "data.noise_model"),
        (lambda cfg: cfg["data"].update(rounds_policy="fixed"), "data.rounds_policy"),
        (
            lambda cfg: cfg["data"]["train_sweep"].update(spacing="logspace"),
            "data.train_sweep.spacing",
        ),
        (lambda cfg: cfg["data"]["train_sweep"].update(count=0), "data.train_sweep.count"),
        (
            lambda cfg: cfg["data"]["train_sweep"].update(start=float("nan")),
            "data.train_sweep.start",
        ),
        (
            lambda cfg: cfg["data"]["train_sweep"].update(start=0.0),
            "data.train_sweep.start",
        ),
        (lambda cfg: cfg["data"].update(total_train_samples=0), "data.total_train_samples"),
        (
            lambda cfg: cfg["data"].update(validation_fraction=1.0),
            "data.validation_fraction",
        ),
        (lambda cfg: cfg["training"].update(batch_size=0), "training.batch_size"),
        (lambda cfg: cfg["training"].update(num_epochs=0), "training.num_epochs"),
        (lambda cfg: cfg["training"].update(peak_lr=float("inf")), "training.peak_lr"),
        (lambda cfg: cfg["training"].update(warmup_steps=-1), "training.warmup_steps"),
        (lambda cfg: cfg["training"].update(focal_gamma=-0.1), "training.focal_gamma"),
        (lambda cfg: cfg["training"].update(focal_alpha=1.1), "training.focal_alpha"),
        (lambda cfg: cfg["evaluation"].update(num_test=0), "evaluation.num_test"),
        (
            lambda cfg: cfg["evaluation"].update(reference_csv=str(Path("missing.csv"))),
            "evaluation.reference_csv",
        ),
        (
            lambda cfg: cfg["evaluation"].update(threshold_pairs=[[3, 5, 7]]),
            "evaluation.threshold_pairs",
        ),
        (
            lambda cfg: cfg["evaluation"].update(threshold_pairs=[[3, 7]]),
            "evaluation.threshold_pairs",
        ),
        (lambda cfg: cfg["paths"].update(result_dir=""), "paths.result_dir"),
        (
            lambda cfg: cfg["paths"].update(baseline_dir=str(Path("missing-baseline"))),
            "paths.baseline_dir",
        ),
    ],
)
def test_load_run_config_rejects_invalid_semantics(
    tmp_path: Path,
    mutation,
    expected_message: str,
) -> None:
    config = valid_config(tmp_path)
    mutation(config)
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match=expected_message):
        load_run_config(config_path)
