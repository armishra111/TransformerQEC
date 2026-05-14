"""Checkpoint IO + listing — local and gs:// transparent.

Two filename conventions handled:
  - Legacy (matches results/legacy/):  transformer_qec_d{D}.pkl
  - Timestamped (Cycle-C onward):      best_d{D}_{YYYYMMDD_HHMMSS}_checkpoint.pkl

For baseline reproduction we keep the legacy filename so eval can drop the
uploaded `gs://.../transformer_qec_d3.pkl` straight in.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

_STAMP_RE = re.compile(r"best_d(\d+)_(\d{8}_\d{6})_checkpoint\.pkl$")
_LEGACY_RE = re.compile(r"transformer_qec_d(\d+)\.pkl$")
_GCS_PREFIX = "gs://"


def _is_gcs(path) -> bool:
    return str(path).startswith(_GCS_PREFIX)


def _gcs_fs():
    import gcsfs
    return gcsfs.GCSFileSystem()


def save_pickle(obj, path) -> None:
    path_s = str(path)
    if _is_gcs(path_s):
        with _gcs_fs().open(path_s, "wb") as f:
            pickle.dump(obj, f)
    else:
        Path(path_s).parent.mkdir(parents=True, exist_ok=True)
        with open(path_s, "wb") as f:
            pickle.dump(obj, f)


def load_pickle(path):
    path_s = str(path)
    if _is_gcs(path_s):
        with _gcs_fs().open(path_s, "rb") as f:
            return pickle.load(f)
    with open(path_s, "rb") as f:
        return pickle.load(f)


def _list_dir(ckpt_dir: str) -> list[str]:
    if _is_gcs(ckpt_dir):
        fs = _gcs_fs()
        paths = fs.glob(ckpt_dir.rstrip("/") + "/*.pkl")
        norm = []
        for p in paths:
            p_s = str(p)
            norm.append(p_s if p_s.startswith(_GCS_PREFIX) else f"{_GCS_PREFIX}{p_s}")
        return norm
    return [str(p) for p in Path(ckpt_dir).glob("*.pkl")]


def discover_checkpoints(ckpt_dir) -> dict[int, list[tuple[str, str]]]:
    """Return {distance: [(stamp_or_'legacy', path), ...]} sorted asc by stamp.

    Legacy filenames sort first (stamp='legacy'), then timestamped files in
    ISO chronological order. `pick_latest` reads the last entry.
    """
    out: dict[int, list[tuple[str, str]]] = {}
    for path in _list_dir(str(ckpt_dir)):
        name = path.rsplit("/", 1)[-1]
        m_stamp = _STAMP_RE.search(name)
        m_legacy = _LEGACY_RE.search(name)
        if m_stamp:
            d = int(m_stamp.group(1))
            out.setdefault(d, []).append((m_stamp.group(2), path))
        elif m_legacy:
            d = int(m_legacy.group(1))
            out.setdefault(d, []).append(("legacy", path))
    for d in out:
        # 'legacy' < any '2025...' string lexicographically — keep legacy first.
        out[d].sort(key=lambda sp: sp[0])
    return out


def pick_latest(ckpt_dir, distance: int, pin: str | None = None
                ) -> tuple[str, str] | None:
    by_d = discover_checkpoints(ckpt_dir)
    candidates = by_d.get(distance, [])
    if not candidates:
        return None
    if pin is not None:
        match = [(s, p) for s, p in candidates if s == pin]
        return match[0] if match else None
    return candidates[-1]


def delete_path(path) -> None:
    path_s = str(path)
    if _is_gcs(path_s):
        try:
            _gcs_fs().rm(path_s)
        except FileNotFoundError:
            pass
    else:
        p = Path(path_s)
        if p.exists():
            p.unlink()
