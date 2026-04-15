from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchCandidate:
    name: str
    config_path: Path
    target_metric: str
    hypothesis: str


_CANDIDATES = {
    "rope-ratio-3-1": ResearchCandidate(
        name="rope-ratio-3-1",
        config_path=Path("configs/experiments/rope_ratio_3_1.yaml"),
        target_metric="transformer_ler",
        hypothesis="Current anisotropic RoPE should remain the baseline reference.",
    ),
    "rope-ratio-1-1": ResearchCandidate(
        name="rope-ratio-1-1",
        config_path=Path("configs/experiments/rope_ratio_1_1.yaml"),
        target_metric="transformer_ler",
        hypothesis="An isotropic split may improve higher-distance generalization.",
    ),
}


def list_candidates() -> list[ResearchCandidate]:
    return list(_CANDIDATES.values())


def get_candidate(name: str) -> ResearchCandidate:
    try:
        return _CANDIDATES[name]
    except KeyError as exc:
        known = ", ".join(_CANDIDATES)
        raise ValueError(f"unknown research candidate: {name!r}; known candidates: {known}") from exc
