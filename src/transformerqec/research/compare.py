from pathlib import Path

from transformerqec.artifacts.io import load_evaluation_rows


def compare_csvs(reference_csv: Path, candidate_csv: Path) -> list[dict[str, float]]:
    reference_rows = load_evaluation_rows(reference_csv)
    candidate_rows = load_evaluation_rows(candidate_csv)
    if len(reference_rows) != len(candidate_rows):
        raise ValueError("comparison CSVs must contain the same number of rows")

    comparisons = []
    for index, (reference, candidate) in enumerate(zip(reference_rows, candidate_rows)):
        if (
            reference["distance"] != candidate["distance"]
            or reference["physical_error_rate"] != candidate["physical_error_rate"]
        ):
            raise ValueError(f"comparison rows are not aligned at index {index}")

        comparisons.append(
            {
                "distance": reference["distance"],
                "physical_error_rate": reference["physical_error_rate"],
                "reference_transformer_ler": reference["transformer_ler"],
                "candidate_transformer_ler": candidate["transformer_ler"],
                "delta_transformer_ler": candidate["transformer_ler"] - reference["transformer_ler"],
            }
        )
    return comparisons
