import math
from pathlib import Path

import numpy as np
import pytest

from transformerqec.baselines.pymatching_decoder import decode_with_pymatching
from transformerqec.evaluation.benchmark import (
    write_benchmark_rows,
    write_threshold_summary,
)
from transformerqec.evaluation.metrics import improvement_pct, logical_error_rate


def test_logical_error_rate_matches_fraction_of_mistakes() -> None:
    assert logical_error_rate([0, 1, 1, 0], [0, 1, 0, 0]) == 0.25


def test_logical_error_rate_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        logical_error_rate([], [])


def test_logical_error_rate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="predictions and labels must have the same length"):
        logical_error_rate([0, 1], [0])


def test_improvement_pct_handles_zero_mwpm_baseline() -> None:
    assert math.isnan(improvement_pct(0.0, 0.1))


def test_improvement_pct_matches_relative_reduction() -> None:
    assert improvement_pct(0.01, 0.009) == pytest.approx(10.0)


def test_write_benchmark_rows_creates_expected_header_and_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "evaluation_results.csv"
    write_benchmark_rows(
        csv_path,
        [
            {
                "d": 3,
                "p": 0.005,
                "mwpm_ler": 0.01,
                "transformer_ler": 0.009,
                "improvement_pct": 10.0,
            },
        ],
    )

    text = csv_path.read_text()

    assert text.splitlines()[0] == "d,p,mwpm_ler,transformer_ler,improvement_pct"
    assert "3,0.005,0.01,0.009,10.0" in text


def test_write_threshold_summary_writes_lines(tmp_path: Path) -> None:
    summary_path = tmp_path / "threshold_summary.txt"

    write_threshold_summary(summary_path, ["line one", "line two"])

    assert summary_path.read_text() == "line one\nline two\n"


def test_decode_with_pymatching_returns_observable_predictions() -> None:
    import stim

    circuit = stim.Circuit(
        """
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    syndromes = np.array([[False], [True]])

    predictions = decode_with_pymatching(circuit, syndromes)

    np.testing.assert_array_equal(predictions, np.array([0, 1], dtype=np.int64))


def test_decode_with_pymatching_rejects_1d_syndromes() -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")

    with pytest.raises(ValueError, match="syndromes must be a 2D array"):
        decode_with_pymatching(circuit, np.array([False, True]))


def test_decode_with_pymatching_rejects_circuit_with_no_observable() -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]")
    syndromes = np.zeros((2, 1), dtype=bool)

    with pytest.raises(ValueError, match="PyMatching returned no observable predictions"):
        decode_with_pymatching(circuit, syndromes)
