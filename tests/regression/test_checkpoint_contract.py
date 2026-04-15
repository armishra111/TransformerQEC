from pathlib import Path

from transformerqec.artifacts.io import load_checkpoint_bundle


ROOT = Path(__file__).resolve().parents[2]


def test_checkpoint_contract() -> None:
    cases = [
        (
            "transformer_qec_d3.pkl",
            3,
            (24, 3),
            11,
            True,
        ),
        (
            "transformer_qec_d5.pkl",
            5,
            (120, 3),
            11,
            True,
        ),
        (
            "transformer_qec_d7.pkl",
            7,
            (336, 3),
            None,
            False,
        ),
    ]

    for filename, distance, shape, epoch, metadata_present in cases:
        bundle = load_checkpoint_bundle(ROOT / "results" / filename)
        assert bundle.config["distance"] == distance
        assert bundle.coords.shape == shape
        assert bundle.params
        assert bundle.metadata["epoch"] == epoch
        if metadata_present:
            assert bundle.metadata["val_loss"] is not None
            assert bundle.metadata["val_acc"] is not None
        else:
            assert bundle.metadata["val_loss"] is None
            assert bundle.metadata["val_acc"] is None
