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
