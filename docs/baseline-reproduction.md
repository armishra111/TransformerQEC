# Baseline Reproduction

Use this procedure when checking the current package-facing baseline entry point.

1. Run `uv sync --extra dev`.
2. Run `uv run transformerqec reproduce-baseline --config configs/laptop/d3-smoke.yaml`.
3. Confirm the command validates the config and prints readiness for the experiment.

The current `reproduce-baseline` CLI is a readiness placeholder: it validates the config and prints readiness, but it does not regenerate artifacts, write `results/runs/`, or compare new outputs against the blessed baseline. The checked-in baseline manifest at `results/baseline/manifest.json` remains the reference artifact contract.

`uv run python scripts/bless_baseline.py` is a maintainer-only baseline acceptance/refresh step. It can overwrite `results/baseline/`, so it is not part of routine reproduction.

For future package-backed research runs, follow the intended shape: add or reuse a config, save outputs under the run artifact tree, and record the comparison against the current blessed baseline before making a claim.
