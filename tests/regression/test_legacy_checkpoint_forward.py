from pathlib import Path

import jax.numpy as jnp

from transformerqec.artifacts.io import load_checkpoint_bundle
from transformerqec.models.transformer import build_model_for_distance


def test_legacy_d3_checkpoint_can_run_forward() -> None:
    bundle = load_checkpoint_bundle(Path("results/transformer_qec_d3.pkl"))
    model = build_model_for_distance(bundle.config)
    logits = model.apply(
        {"params": bundle.params},
        jnp.zeros((2, bundle.config["seq_len"]), dtype=jnp.float32),
        jnp.array([0.005, 0.01], dtype=jnp.float32),
        jnp.asarray(bundle.coords),
    )
    assert logits.shape == (2, 2)
