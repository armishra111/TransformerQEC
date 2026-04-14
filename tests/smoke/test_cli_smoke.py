from pathlib import Path

from typer.testing import CliRunner

from transformerqec.cli import app

runner = CliRunner()


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
            str(Path("configs/laptop/d3-smoke.yaml")),
        ],
    )
    assert result.exit_code == 0
