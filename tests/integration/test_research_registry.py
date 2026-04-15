from pathlib import Path

import pytest

from transformerqec.config.io import load_run_config
from transformerqec.research.compare import compare_csvs
from transformerqec.research.registry import get_candidate, list_candidates


def test_rope_ratio_candidates_are_registered() -> None:
    names = [candidate.name for candidate in list_candidates()]
    assert "rope-ratio-3-1" in names
    assert "rope-ratio-1-1" in names


def test_candidate_lookup_returns_target_metric() -> None:
    candidate = get_candidate("rope-ratio-3-1")
    assert candidate.target_metric == "transformer_ler"


def test_registered_candidate_configs_exist_and_load() -> None:
    for candidate in list_candidates():
        assert candidate.config_path.exists()
        config = load_run_config(candidate.config_path)
        assert config.experiment_name == candidate.name


def test_unknown_candidate_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown research candidate"):
        get_candidate("missing-candidate")


def test_compare_csvs_returns_transformer_ler_delta(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    candidate_csv = tmp_path / "candidate.csv"
    reference_csv.write_text(
        "d,p,mwpm_ler,transformer_ler,improvement_pct\n"
        "3,0.005,0.02,0.01,50.0\n"
    )
    candidate_csv.write_text(
        "d,p,mwpm_ler,transformer_ler,improvement_pct\n"
        "3,0.005,0.02,0.008,60.0\n"
    )

    assert compare_csvs(reference_csv, candidate_csv) == [
        {
            "distance": 3,
            "physical_error_rate": 0.005,
            "reference_transformer_ler": 0.01,
            "candidate_transformer_ler": 0.008,
            "delta_transformer_ler": pytest.approx(-0.002),
        }
    ]


def test_compare_csvs_rejects_misaligned_rows(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    candidate_csv = tmp_path / "candidate.csv"
    reference_csv.write_text(
        "d,p,mwpm_ler,transformer_ler,improvement_pct\n"
        "3,0.005,0.02,0.01,50.0\n"
    )
    candidate_csv.write_text(
        "d,p,mwpm_ler,transformer_ler,improvement_pct\n"
        "3,0.01,0.02,0.008,60.0\n"
    )

    with pytest.raises(ValueError, match="comparison rows are not aligned"):
        compare_csvs(reference_csv, candidate_csv)
