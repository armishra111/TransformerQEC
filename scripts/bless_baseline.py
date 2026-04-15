from pathlib import Path
import shutil

from transformerqec.artifacts.manifest import write_manifest


ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_NAMES = [
    "transformer_qec_d3.pkl",
    "transformer_qec_d5.pkl",
    "transformer_qec_d7.pkl",
    "evaluation_results.csv",
    "threshold_estimates.txt",
    "logical_error_rates.png",
    "transformer_vs_mwpm.png",
]


def bless_baseline(root: Path = ROOT) -> None:
    source = root / "results"
    dest = root / "results" / "baseline"
    source_paths = [source / relative_name for relative_name in ARTIFACT_NAMES]
    missing_paths = [path for path in source_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"baseline source files do not exist: {missing}")

    dest.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for source_path in source_paths:
        dest_path = dest / source_path.name
        shutil.copy2(source_path, dest_path)
        copied_files.append(dest_path)

    write_manifest(dest / "manifest.json", copied_files)


if __name__ == "__main__":
    bless_baseline()
