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
    assert "1.3M parameter" not in text
    assert "bfloat16" not in text
    assert "10M synthetic syndromes" not in text
    assert "learned Y-error" not in text
    assert "generalized topological homology" not in text
    assert "outperform classical graph-matching decoders" not in text
    assert "Built with JAX on TPU" not in text
    assert "Built with JAX. Synthetic data generated with STIM." in text


def test_baseline_docs_match_current_cli_behavior() -> None:
    text = Path("docs/baseline-reproduction.md").read_text()
    assert "validates the config and prints readiness" in text
    assert "does not regenerate artifacts" in text
    assert "maintainer-only baseline acceptance/refresh step" in text
    assert "routine reproduction" in text


def test_notebooks_are_archived() -> None:
    text = Path("notebooks/README.md").read_text()
    assert "have been moved under `notebooks/archive/`" in text
    assert "Once package parity is in place, move them" not in text
    assert Path("notebooks/archive/01_data_exploration.ipynb").exists()
    assert Path("notebooks/archive/02_model_and_training.ipynb").exists()
    assert Path("notebooks/archive/03_evaluation.ipynb").exists()
    assert not Path("notebooks/01_data_exploration.ipynb").exists()
    assert not Path("notebooks/02_model_and_training.ipynb").exists()
    assert not Path("notebooks/03_evaluation.ipynb").exists()


def test_architecture_does_not_claim_research_package_exists() -> None:
    text = Path("docs/architecture.md").read_text()
    assert (
        "`transformerqec.training`, `transformerqec.evaluation`, and "
        "`transformerqec.research` own"
    ) not in text
    assert "A research registry layer is planned" in text
