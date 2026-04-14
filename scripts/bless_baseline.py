from pathlib import Path
import shutil

from transformerqec.artifacts.manifest import write_manifest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results"
DEST = ROOT / "results" / "baseline"

FILES = [
    "transformer_qec_d3.pkl",
    "transformer_qec_d5.pkl",
    "transformer_qec_d7.pkl",
    "evaluation_results.csv",
    "threshold_estimates.txt",
    "logical_error_rates.png",
    "transformer_vs_mwpm.png",
]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for relative_name in FILES:
        source_path = SOURCE / relative_name
        dest_path = DEST / relative_name
        shutil.copy2(source_path, dest_path)
        copied_files.append(dest_path)

    write_manifest(DEST / "manifest.json", copied_files)


if __name__ == "__main__":
    main()
