from pathlib import Path

from transformerqec.artifacts.io import load_checkpoint_bundle


ROOT = Path(__file__).resolve().parents[2]


def test_d3_checkpoint_contract() -> None:
    bundle = load_checkpoint_bundle(ROOT / "results" / "transformer_qec_d3.pkl")
    assert bundle.config["distance"] == 3
    assert bundle.coords.shape == (24, 3)
    assert bundle.metadata["epoch"] == 11


def test_d7_checkpoint_allows_missing_training_metadata() -> None:
    bundle = load_checkpoint_bundle(ROOT / "results" / "transformer_qec_d7.pkl")
    assert bundle.config["distance"] == 7
    assert bundle.metadata["epoch"] is None
    assert bundle.metadata["val_loss"] is None
