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
