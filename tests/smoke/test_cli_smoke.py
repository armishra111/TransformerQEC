from pathlib import Path

import pytest
from typer.testing import CliRunner

from transformerqec.cli import app

runner = CliRunner()
SMOKE_CONFIG = "configs/laptop/d3-smoke.yaml"


def test_help_mentions_reproduce_baseline() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "reproduce-baseline" in result.stdout


def test_reproduce_baseline_accepts_checked_in_config() -> None:
    result = runner.invoke(
        app,
        [
            "reproduce-baseline",
            "--config",
            str(Path(SMOKE_CONFIG)),
        ],
    )
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("command", "expected_output"),
    [
        ("generate", "Loaded data generation config:"),
        ("train", "Loaded training config:"),
        ("eval", "Loaded evaluation config:"),
        ("reproduce-baseline", "Baseline reproduction ready for d3-smoke"),
        ("benchmark", "Benchmark config accepted:"),
        ("infer", "Inference config accepted:"),
    ],
)
def test_commands_accept_checked_in_config(command: str, expected_output: str) -> None:
    result = runner.invoke(app, [command, "--config", SMOKE_CONFIG])
    assert result.exit_code == 0
    assert expected_output in result.stdout


def test_command_requires_config_option() -> None:
    result = runner.invoke(app, ["reproduce-baseline"])
    assert result.exit_code != 0
    assert "--config" in result.output


def test_reproduce_baseline_rejects_invalid_config_before_readiness(tmp_path: Path) -> None:
    reference_csv = tmp_path / "evaluation_results.csv"
    reference_csv.write_text("d,p,mwpm_ler,transformer_ler,improvement_pct\n")
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        f"""
experiment_name: invalid
model:
  d_model: 0
  num_heads: 4
  num_layers_by_distance: {{3: 4}}
  ffn_dim_by_distance: {{3: 1024}}
  pos_encoding: rope
  rope_spatial_ratio: 3
  rope_temporal_ratio: 1
data:
  distances: [3]
  noise_model: phenomenological
  rounds_policy: distance
  train_sweep: {{start: 0.005, stop: 0.01, count: 2, spacing: geomspace}}
  eval_sweep: {{start: 0.005, stop: 0.01, count: 2, spacing: geomspace}}
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
  reference_csv: {reference_csv}
  threshold_pairs: [[3, 3]]
paths:
  result_dir: {tmp_path / "results"}
  baseline_dir: {baseline_dir}
  run_dir: {tmp_path / "runs"}
""".strip()
    )

    result = runner.invoke(app, ["reproduce-baseline", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "Baseline reproduction ready" not in result.stdout
