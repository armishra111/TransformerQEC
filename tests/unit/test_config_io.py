from pathlib import Path

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
