from importlib.metadata import version

from typer.testing import CliRunner

from transformerqec import __version__
from transformerqec.cli import app


runner = CliRunner()


def test_package_version_matches_metadata() -> None:
    assert __version__ == version("transformerqec")


def test_cli_app_has_expected_name() -> None:
    assert app.info.name == "transformerqec"


def test_cli_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert (
        "TransformerQEC decoder library CLI" in result.stdout
        or "transformerqec" in result.stdout
    )
