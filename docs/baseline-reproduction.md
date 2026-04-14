# Baseline Reproduction

Use this procedure when refreshing or checking the package-facing baseline.

1. Run `uv sync --extra dev`.
2. Run `uv run python scripts/bless_baseline.py`.
3. Run `uv run transformerqec reproduce-baseline --config configs/laptop/d3-smoke.yaml`.
4. Compare outputs under `results/runs/` against `results/baseline/manifest.json`.

The checked-in baseline manifest is the reference for smoke-scale reproducibility. A reproduction is expected to identify the config used, the artifacts produced, and any metric differences from the blessed artifact set.

For larger research runs, follow the same shape: add or reuse a config, save outputs under the run artifact tree, and record the comparison against the current blessed baseline before making a claim.
