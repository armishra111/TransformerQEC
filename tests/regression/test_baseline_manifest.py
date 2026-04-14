import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from transformerqec.artifacts.manifest import write_manifest


BASELINE = Path("results/baseline")
ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FILES = {
    "transformer_qec_d3.pkl",
    "transformer_qec_d5.pkl",
    "transformer_qec_d7.pkl",
    "evaluation_results.csv",
    "threshold_estimates.txt",
    "logical_error_rates.png",
    "transformer_vs_mwpm.png",
}


def load_bless_baseline_script():
    spec = importlib.util.spec_from_file_location(
        "bless_baseline_script",
        ROOT / "scripts" / "bless_baseline.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blessed_manifest_tracks_existing_files() -> None:
    manifest = json.loads((BASELINE / "manifest.json").read_text())
    tracked = {item["relative_path"] for item in manifest["files"]}
    assert tracked == EXPECTED_FILES
    assert {path.name for path in BASELINE.iterdir()} == EXPECTED_FILES | {"manifest.json"}

    for item in manifest["files"]:
        baseline_file = BASELINE / item["relative_path"]
        assert baseline_file.exists()
        assert item["sha256"] == hashlib.sha256(baseline_file.read_bytes()).hexdigest()
        assert item["size_bytes"] == baseline_file.stat().st_size


def test_manifest_rejects_empty_file_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one file"):
        write_manifest(tmp_path / "manifest.json", [])


def test_manifest_rejects_missing_input_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="missing.txt"):
        write_manifest(tmp_path / "manifest.json", [missing_file])


def test_bless_baseline_missing_source_does_not_mutate_existing_baseline(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    baseline = results / "baseline"
    results.mkdir()
    baseline.mkdir()
    existing_baseline = baseline / "transformer_qec_d3.pkl"
    existing_baseline.write_text("old baseline")

    missing_name = "threshold_estimates.txt"
    for artifact_name in EXPECTED_FILES - {missing_name}:
        (results / artifact_name).write_text(f"source {artifact_name}")

    with pytest.raises(FileNotFoundError, match=missing_name):
        baseline_script = load_bless_baseline_script()
        baseline_script.bless_baseline(tmp_path)

    assert existing_baseline.read_text() == "old baseline"
