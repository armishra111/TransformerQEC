import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(output_path: Path, files: list[Path]) -> None:
    if not files:
        raise ValueError("manifest requires at least one file")

    missing_files = [path for path in files if not path.is_file()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"manifest input files do not exist: {missing}")

    payload = {
        "baseline_name": "current",
        "files": [
            {
                "relative_path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(files, key=lambda item: item.name)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
