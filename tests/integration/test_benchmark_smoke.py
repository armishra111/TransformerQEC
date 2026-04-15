import math
from pathlib import Path

import numpy as np
import pytest
import transformerqec.baselines.pymatching_decoder as pymatching_decoder

from transformerqec.baselines.pymatching_decoder import decode_with_pymatching
from transformerqec.codes.surface_code import make_rotated_memory_z_circuit
from transformerqec.evaluation.benchmark import (
    write_benchmark_rows,
    write_threshold_summary,
)
from transformerqec.evaluation.metrics import improvement_pct, logical_error_rate


def test_logical_error_rate_matches_fraction_of_mistakes() -> None:
    assert logical_error_rate([0, 1, 1, 0], [0, 1, 0, 0]) == 0.25


def test_logical_error_rate_accepts_bool_and_exact_binary_numbers() -> None:
    assert logical_error_rate([False, 1.0, True], [0, 1, 0.0]) == pytest.approx(1 / 3)


def test_logical_error_rate_accepts_numpy_int64_decoder_outputs() -> None:
    predictions = np.array([0, 1], dtype=np.int64)
    labels = np.array([0, 1], dtype=np.int64)

    assert logical_error_rate(predictions, labels) == 0.0


def test_logical_error_rate_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        logical_error_rate([], [])


def test_logical_error_rate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="predictions and labels must have the same length"):
        logical_error_rate([0, 1], [0])


@pytest.mark.parametrize(
    ("predictions", "labels"),
    [
        ([0.2], [0]),
        ([2], [0]),
        ([np.nan], [0]),
        ([np.inf], [0]),
        ([0], [0.7]),
        ([0], [-1]),
        ([0], [np.nan]),
        ([0], [np.inf]),
    ],
)
def test_logical_error_rate_rejects_non_binary_or_non_finite_values(
    predictions,
    labels,
) -> None:
    with pytest.raises(ValueError, match="predictions and labels must contain only finite binary values"):
        logical_error_rate(predictions, labels)


def test_improvement_pct_handles_zero_mwpm_baseline() -> None:
    assert math.isnan(improvement_pct(0.0, 0.1))


def test_improvement_pct_matches_relative_reduction() -> None:
    assert improvement_pct(0.01, 0.009) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("mwpm_ler", "transformer_ler"),
    [
        (math.nan, 0.1),
        (math.inf, 0.1),
        (0.1, math.nan),
        (0.1, math.inf),
        (-0.1, 0.1),
        (1.1, 0.1),
        (0.1, -0.1),
        (0.1, 1.1),
    ],
)
def test_improvement_pct_rejects_invalid_lers(mwpm_ler: float, transformer_ler: float) -> None:
    with pytest.raises(ValueError, match="LER values must be finite values in \\[0, 1\\]"):
        improvement_pct(mwpm_ler, transformer_ler)


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


def test_write_benchmark_rows_rejects_missing_field(tmp_path: Path) -> None:
    csv_path = tmp_path / "evaluation_results.csv"

    with pytest.raises(ValueError, match="benchmark row 0 is missing required fields"):
        write_benchmark_rows(
            csv_path,
            [{"d": 3, "p": 0.005, "mwpm_ler": 0.01, "transformer_ler": 0.009}],
        )


def test_write_benchmark_rows_rejects_none_required_field(tmp_path: Path) -> None:
    csv_path = tmp_path / "evaluation_results.csv"

    with pytest.raises(ValueError, match="benchmark row 0 is missing required fields: mwpm_ler"):
        write_benchmark_rows(
            csv_path,
            [
                {
                    "d": 3,
                    "p": 0.005,
                    "mwpm_ler": None,
                    "transformer_ler": 0.009,
                    "improvement_pct": 10.0,
                },
            ],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("d", 0),
        ("d", 3.5),
        ("p", -0.1),
        ("p", math.nan),
        ("mwpm_ler", 2.0),
        ("transformer_ler", -0.3),
        ("improvement_pct", math.inf),
    ],
)
def test_write_benchmark_rows_rejects_impossible_metric_values(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    csv_path = tmp_path / "evaluation_results.csv"
    row = {
        "d": 3,
        "p": 0.005,
        "mwpm_ler": 0.01,
        "transformer_ler": 0.009,
        "improvement_pct": 10.0,
    }
    row[field] = value

    with pytest.raises(ValueError, match=f"benchmark row 0 has invalid {field}"):
        write_benchmark_rows(csv_path, [row])


def test_write_benchmark_rows_allows_nan_improvement_pct(tmp_path: Path) -> None:
    csv_path = tmp_path / "evaluation_results.csv"

    write_benchmark_rows(
        csv_path,
        [
            {
                "d": 3,
                "p": 0.005,
                "mwpm_ler": 0.0,
                "transformer_ler": 0.0,
                "improvement_pct": math.nan,
            },
        ],
    )

    assert "nan" in csv_path.read_text()


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


def test_decode_with_pymatching_reuses_cached_matching_for_equivalent_circuits() -> None:
    import stim

    circuit = stim.Circuit(
        """
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    equivalent_circuit = stim.Circuit(str(circuit))

    first = pymatching_decoder._get_matching(circuit)
    second = pymatching_decoder._get_matching(equivalent_circuit)

    assert first is second


def test_decode_with_pymatching_accepts_binary_float_syndromes() -> None:
    import stim

    circuit = stim.Circuit(
        """
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    syndromes = np.array([[0.0], [1.0]], dtype=np.float32)

    predictions = decode_with_pymatching(circuit, syndromes)

    np.testing.assert_array_equal(predictions, np.array([0, 1], dtype=np.int64))


def test_decode_with_pymatching_rejects_1d_syndromes() -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")

    with pytest.raises(ValueError, match="syndromes must be a 2D array"):
        decode_with_pymatching(circuit, np.array([False, True]))


@pytest.mark.parametrize(
    "syndromes",
    [
        np.array([[0.2]], dtype=np.float32),
        np.array([[-1.0]], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[np.inf]], dtype=np.float32),
    ],
)
def test_decode_with_pymatching_rejects_invalid_syndrome_values(
    syndromes: np.ndarray,
) -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")

    with pytest.raises(ValueError, match="syndromes must contain only finite binary values"):
        decode_with_pymatching(circuit, syndromes)


def test_decode_with_pymatching_rejects_circuit_with_no_observable() -> None:
    import stim

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]")
    syndromes = np.zeros((2, 1), dtype=bool)

    with pytest.raises(ValueError, match="PyMatching returned no observable predictions"):
        decode_with_pymatching(circuit, syndromes)


def test_decode_with_pymatching_handles_generated_surface_code_circuit() -> None:
    circuit = make_rotated_memory_z_circuit(
        distance=3,
        physical_error_rate=0.001,
        rounds=1,
    )
    syndromes = np.zeros((2, circuit.num_detectors), dtype=bool)

    predictions = decode_with_pymatching(circuit, syndromes)

    np.testing.assert_array_equal(predictions, np.zeros(2, dtype=np.int64))
