from importlib.metadata import version

from transformerqec import __version__
from transformerqec.cli import app


def test_package_version_matches_metadata() -> None:
    assert __version__ == version("transformerqec")


def test_cli_app_has_expected_name() -> None:
    assert app.info.name == "transformerqec"
