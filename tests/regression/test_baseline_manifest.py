import hashlib
import json
from pathlib import Path

import pytest

from transformerqec.artifacts.manifest import write_manifest


BASELINE = Path("results/baseline")
EXPECTED_FILES = {
    "transformer_qec_d3.pkl",
    "transformer_qec_d5.pkl",
    "transformer_qec_d7.pkl",
    "evaluation_results.csv",
    "threshold_estimates.txt",
    "logical_error_rates.png",
    "transformer_vs_mwpm.png",
}


def test_blessed_manifest_tracks_existing_files() -> None:
    manifest = json.loads((BASELINE / "manifest.json").read_text())
    tracked = {item["relative_path"] for item in manifest["files"]}
    assert tracked == EXPECTED_FILES

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
