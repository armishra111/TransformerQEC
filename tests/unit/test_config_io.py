from pathlib import Path

import pytest

from transformerqec.config.io import load_run_config, materialize_sweep


def test_current_config_materializes_expected_training_grid() -> None:
    config = load_run_config(Path("configs/baseline/current.yaml"))
    train_grid = materialize_sweep(config.data.train_sweep)
    assert len(train_grid) == 20
    assert round(train_grid[0], 6) == 0.002
    assert round(train_grid[-1], 6) == 0.017


def test_smoke_config_stays_small() -> None:
    config = load_run_config(Path("configs/laptop/d3-smoke.yaml"))
    assert config.data.distances == [3]
    assert config.training.batch_size == 64
    assert config.evaluation.num_test == 2048


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
  reference_csv: results/evaluation_results.csv
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
