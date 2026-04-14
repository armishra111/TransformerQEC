from pathlib import Path

import jax.numpy as jnp
import pytest

from transformerqec.artifacts.io import load_checkpoint_bundle
from transformerqec.models.transformer import build_model_for_distance


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "transformer_qec_d3.pkl",
        "transformer_qec_d5.pkl",
        "transformer_qec_d7.pkl",
    ],
)
def test_legacy_checkpoint_can_run_forward(checkpoint_name: str) -> None:
    bundle = load_checkpoint_bundle(Path("results") / checkpoint_name)
    model = build_model_for_distance(bundle.config)
    logits = model.apply(
        {"params": bundle.params},
        jnp.zeros((1, bundle.config["seq_len"]), dtype=jnp.float32),
        jnp.array([0.005], dtype=jnp.float32),
        jnp.asarray(bundle.coords),
    )
    assert logits.shape == (1, 2)
