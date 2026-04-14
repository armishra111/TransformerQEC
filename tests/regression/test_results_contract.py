from pathlib import Path

from transformerqec.artifacts.io import load_evaluation_rows


ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_csv_schema_is_stable() -> None:
    rows = load_evaluation_rows(ROOT / "results" / "evaluation_results.csv")
    assert len(rows) == 30
    assert rows[0].keys() == {
        "distance",
        "physical_error_rate",
        "mwpm_ler",
        "transformer_ler",
        "improvement_pct",
    }
    assert rows[-1] == {
        "distance": 7,
        "physical_error_rate": 0.01,
        "mwpm_ler": 0.00036,
        "transformer_ler": 0.0016100000000000003,
        "improvement_pct": -347.2222222222223,
    }
    assert [row["distance"] for row in rows[:10]] == [3] * 10
    assert [row["distance"] for row in rows[10:20]] == [5] * 10
    assert [row["distance"] for row in rows[20:]] == [7] * 10
