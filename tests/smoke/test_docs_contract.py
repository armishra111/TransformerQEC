from pathlib import Path


def test_required_docs_exist() -> None:
    assert Path("docs/architecture.md").exists()
    assert Path("docs/baseline-reproduction.md").exists()
    assert Path("docs/research-landscape.md").exists()
    assert Path("docs/research-benchmark-contract.md").exists()


def test_readme_mentions_uv_and_baseline_command() -> None:
    text = Path("README.md").read_text()
    assert "uv sync" in text
    assert "uv run transformerqec reproduce-baseline" in text
    assert "head_dim=32" in text
    assert "6000000 total training samples" in text


def test_baseline_docs_match_current_cli_behavior() -> None:
    text = Path("docs/baseline-reproduction.md").read_text()
    assert "validates the config and prints readiness" in text
    assert "does not regenerate artifacts" in text
    assert "maintainer-only baseline acceptance/refresh step" in text
    assert "routine reproduction" in text


def test_notebooks_are_archived() -> None:
    assert Path("notebooks/archive/01_data_exploration.ipynb").exists()
    assert Path("notebooks/archive/02_model_and_training.ipynb").exists()
    assert Path("notebooks/archive/03_evaluation.ipynb").exists()
    assert not Path("notebooks/01_data_exploration.ipynb").exists()
    assert not Path("notebooks/02_model_and_training.ipynb").exists()
    assert not Path("notebooks/03_evaluation.ipynb").exists()
