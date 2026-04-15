# TransformerQEC Library Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor TransformerQEC into a UV-managed decoder library that preserves the current repository artifacts as a blessed baseline, replaces notebook-owned logic with importable modules, and creates a verified path for post-baseline research ablations.

**Architecture:** Freeze the current artifact contract first, because the existing checkpoints and CSV are the only stable truth we have today. Then extract the package in dependency order: package skeleton, artifact loaders, config schema, circuit/data utilities, RoPE/model code, training, evaluation, CLI, docs, and the first research harness. Keep the original results as the first blessed baseline so every later claim is anchored to files that already exist in the repo.

**Tech Stack:** Python 3.11+, UV, JAX, Flax, Optax, STIM, PyMatching, NumPy, Matplotlib, PyYAML, Typer, pytest

---

## Scope Check

This stays as one implementation plan because the work is serial, not independent:

1. baseline protection has to land before the refactor,
2. the package layers depend on each other in a clear order,
3. the research harness is only meaningful after the baseline contract exists.

## File Map

**Modify**

- `.gitignore` - stop ignoring `tests/`; ignore only generated runtime outputs.
- `README.md` - add UV quickstart, architecture summary, validation standard, and doc links.
- `notebooks/01_data_exploration.ipynb` - archive after package parity is established.
- `notebooks/02_model_and_training.ipynb` - archive after package parity is established.
- `notebooks/03_evaluation.ipynb` - archive after package parity is established.

**Create**

- `pyproject.toml`
- `src/transformerqec/__init__.py`
- `src/transformerqec/cli.py`
- `src/transformerqec/config/__init__.py`
- `src/transformerqec/config/schema.py`
- `src/transformerqec/config/io.py`
- `src/transformerqec/codes/__init__.py`
- `src/transformerqec/codes/surface_code.py`
- `src/transformerqec/data/__init__.py`
- `src/transformerqec/data/sampling.py`
- `src/transformerqec/models/__init__.py`
- `src/transformerqec/models/rope.py`
- `src/transformerqec/models/transformer.py`
- `src/transformerqec/training/__init__.py`
- `src/transformerqec/training/losses.py`
- `src/transformerqec/training/state.py`
- `src/transformerqec/training/loop.py`
- `src/transformerqec/baselines/__init__.py`
- `src/transformerqec/baselines/pymatching_decoder.py`
- `src/transformerqec/evaluation/__init__.py`
- `src/transformerqec/evaluation/metrics.py`
- `src/transformerqec/evaluation/benchmark.py`
- `src/transformerqec/artifacts/__init__.py`
- `src/transformerqec/artifacts/io.py`
- `src/transformerqec/artifacts/manifest.py`
- `src/transformerqec/research/__init__.py`
- `src/transformerqec/research/registry.py`
- `src/transformerqec/research/compare.py`
- `configs/baseline/current.yaml`
- `configs/laptop/d3-smoke.yaml`
- `configs/experiments/rope_ratio_3_1.yaml`
- `configs/experiments/rope_ratio_1_1.yaml`
- `tests/unit/test_package_smoke.py`
- `tests/unit/test_config_io.py`
- `tests/unit/test_surface_code.py`
- `tests/unit/test_rope.py`
- `tests/unit/test_transformer_model.py`
- `tests/unit/test_focal_loss.py`
- `tests/integration/test_dataset_sampling.py`
- `tests/integration/test_training_smoke.py`
- `tests/integration/test_benchmark_smoke.py`
- `tests/integration/test_research_registry.py`
- `tests/regression/test_checkpoint_contract.py`
- `tests/regression/test_results_contract.py`
- `tests/regression/test_legacy_checkpoint_forward.py`
- `tests/regression/test_baseline_manifest.py`
- `tests/smoke/test_cli_smoke.py`
- `tests/smoke/test_docs_contract.py`
- `scripts/bless_baseline.py`
- `docs/architecture.md`
- `docs/baseline-reproduction.md`
- `docs/research-landscape.md`
- `docs/research-benchmark-contract.md`
- `notebooks/README.md`
- `results/baseline/manifest.json`

### Task 1: Bootstrap the UV package and tracked test layout

**Files:**
- Modify: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/transformerqec/__init__.py`
- Create: `src/transformerqec/cli.py`
- Create: `tests/unit/test_package_smoke.py`

- [ ] **Step 1: Write the failing package smoke test**

```python
from importlib.metadata import version

from transformerqec import __version__
from transformerqec.cli import app


def test_package_version_matches_metadata() -> None:
    assert __version__ == version("transformerqec")


def test_cli_app_has_expected_name() -> None:
    assert app.info.name == "transformerqec"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_package_smoke.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'transformerqec'`

- [ ] **Step 3: Write the minimal package implementation**

`pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "transformerqec"
version = "0.1.0"
description = "Reusable decoder library extracted from the TransformerQEC notebooks"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "flax>=0.10",
  "jax>=0.7",
  "matplotlib>=3.9",
  "numpy>=2.1",
  "optax>=0.2",
  "pymatching>=2.3",
  "pyyaml>=6.0",
  "stim>=1.14",
  "typer>=0.16",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "ruff>=0.11", "nbformat>=5.10"]

[project.scripts]
transformerqec = "transformerqec.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`src/transformerqec/__init__.py`

```python
from importlib.metadata import version

__all__ = ["__version__"]

__version__ = version("transformerqec")
```

`src/transformerqec/cli.py`

```python
import typer

app = typer.Typer(name="transformerqec", help="TransformerQEC decoder library CLI")
```

`.gitignore`

```gitignore
.venv/
__pycache__/
.pytest_cache/
results/runs/
.claude
```

- [ ] **Step 4: Sync the environment and rerun the smoke test**

Run: `uv sync --extra dev && uv run pytest tests/unit/test_package_smoke.py -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml src/transformerqec/__init__.py src/transformerqec/cli.py tests/unit/test_package_smoke.py
git commit -m "feat: bootstrap transformerqec package skeleton"
```

### Task 2: Freeze the current artifact contract before refactoring

**Files:**
- Create: `src/transformerqec/artifacts/__init__.py`
- Create: `src/transformerqec/artifacts/io.py`
- Create: `tests/regression/test_checkpoint_contract.py`
- Create: `tests/regression/test_results_contract.py`

- [ ] **Step 1: Write failing regression tests for the existing results**

```python
from pathlib import Path

from transformerqec.artifacts.io import load_checkpoint_bundle, load_evaluation_rows

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


def test_evaluation_csv_schema_is_stable() -> None:
    rows = load_evaluation_rows(ROOT / "results" / "evaluation_results.csv")
    assert rows[0].keys() == {
        "distance",
        "physical_error_rate",
        "mwpm_ler",
        "transformer_ler",
        "improvement_pct",
    }
    assert rows[-1]["distance"] == 7
    assert rows[-1]["physical_error_rate"] == 0.01
```

- [ ] **Step 2: Run the regression tests to verify they fail**

Run: `uv run pytest tests/regression/test_checkpoint_contract.py tests/regression/test_results_contract.py -q`

Expected: FAIL with `ImportError` for `transformerqec.artifacts.io`

- [ ] **Step 3: Implement artifact loaders without changing the underlying files**

`src/transformerqec/artifacts/io.py`

```python
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CheckpointBundle:
    config: dict[str, Any]
    coords: np.ndarray
    params: dict[str, Any]
    metadata: dict[str, Any | None]


def load_checkpoint_bundle(path: Path) -> CheckpointBundle:
    raw = pickle.loads(path.read_bytes())
    return CheckpointBundle(
        config=dict(raw["config"]),
        coords=np.asarray(raw["coords"], dtype=np.float32),
        params=dict(raw["params"]),
        metadata={
            "epoch": raw.get("epoch"),
            "val_loss": raw.get("val_loss"),
            "val_acc": raw.get("val_acc"),
        },
    )


def load_evaluation_rows(path: Path) -> list[dict[str, float]]:
    normalized_rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for raw_row in reader:
            normalized_rows.append(
                {
                    "distance": int(raw_row["d"]),
                    "physical_error_rate": float(raw_row["p"]),
                    "mwpm_ler": float(raw_row["mwpm_ler"]),
                    "transformer_ler": float(raw_row["transformer_ler"]),
                    "improvement_pct": float(raw_row["improvement_pct"])
                    if raw_row["improvement_pct"] != "nan"
                    else float("nan"),
                }
            )
    return normalized_rows
```

`src/transformerqec/artifacts/__init__.py`

```python
from transformerqec.artifacts.io import CheckpointBundle, load_checkpoint_bundle, load_evaluation_rows

__all__ = ["CheckpointBundle", "load_checkpoint_bundle", "load_evaluation_rows"]
```

- [ ] **Step 4: Rerun the regression tests**

Run: `uv run pytest tests/regression/test_checkpoint_contract.py tests/regression/test_results_contract.py -q`

Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/artifacts/__init__.py src/transformerqec/artifacts/io.py tests/regression/test_checkpoint_contract.py tests/regression/test_results_contract.py
git commit -m "test: freeze current checkpoint and csv contracts"
```

### Task 3: Add the config schema and the first two checked-in configs

**Files:**
- Create: `src/transformerqec/config/__init__.py`
- Create: `src/transformerqec/config/schema.py`
- Create: `src/transformerqec/config/io.py`
- Create: `configs/baseline/current.yaml`
- Create: `configs/laptop/d3-smoke.yaml`
- Create: `tests/unit/test_config_io.py`

- [ ] **Step 1: Write failing config tests**

```python
from pathlib import Path

from transformerqec.config.io import load_run_config, materialize_sweep


def test_current_config_materializes_expected_training_grid() -> None:
    config = load_run_config(Path("configs/baseline/current.yaml"))
    train_grid = materialize_sweep(config.data.train_sweep)
    assert len(train_grid) == 20
    assert round(train_grid[0], 6) == 0.002
    assert round(train_grid[-1], 6) == 0.017


def test_smoke_config_stays_small() -> None:
    config = load_run_config(Path("configs/laptop/d3-smoke.yaml"))
    assert config.data.distances == [3]
    assert config.training.batch_size == 64
    assert config.evaluation.num_test == 2048
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_io.py -q`

Expected: FAIL with `ImportError` for `transformerqec.config.io`

- [ ] **Step 3: Implement the config dataclasses and YAML loaders**

`src/transformerqec/config/schema.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SweepConfig:
    start: float
    stop: float
    count: int
    spacing: str = "geomspace"


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    num_heads: int
    num_layers_by_distance: dict[int, int]
    ffn_dim_by_distance: dict[int, int]
    pos_encoding: str
    rope_spatial_ratio: int
    rope_temporal_ratio: int


@dataclass(frozen=True)
class DataConfig:
    distances: list[int]
    noise_model: str
    rounds_policy: str
    train_sweep: SweepConfig
    eval_sweep: SweepConfig
    total_train_samples: int
    validation_fraction: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    num_epochs: int
    peak_lr: float
    warmup_steps: int
    focal_gamma: float
    focal_alpha: float
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    num_test: int
    reference_csv: str
    threshold_pairs: list[list[int]]


@dataclass(frozen=True)
class PathsConfig:
    result_dir: str
    baseline_dir: str
    run_dir: str


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
```

`src/transformerqec/config/io.py`

```python
from pathlib import Path

import numpy as np
import yaml

from transformerqec.config.schema import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
    RunConfig,
    SweepConfig,
    TrainingConfig,
)


def materialize_sweep(sweep: SweepConfig) -> list[float]:
    if sweep.spacing == "geomspace":
        return np.geomspace(sweep.start, sweep.stop, sweep.count).tolist()
    return np.linspace(sweep.start, sweep.stop, sweep.count).tolist()


def load_run_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text())
    return RunConfig(
        experiment_name=raw["experiment_name"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(
            distances=raw["data"]["distances"],
            noise_model=raw["data"]["noise_model"],
            rounds_policy=raw["data"]["rounds_policy"],
            train_sweep=SweepConfig(**raw["data"]["train_sweep"]),
            eval_sweep=SweepConfig(**raw["data"]["eval_sweep"]),
            total_train_samples=raw["data"]["total_train_samples"],
            validation_fraction=raw["data"]["validation_fraction"],
        ),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        paths=PathsConfig(**raw["paths"]),
    )
```

`configs/baseline/current.yaml`

```yaml
experiment_name: current-baseline
model:
  d_model: 128
  num_heads: 4
  num_layers_by_distance: {3: 4, 5: 6, 7: 6}
  ffn_dim_by_distance: {3: 1024, 5: 512, 7: 512}
  pos_encoding: rope
  rope_spatial_ratio: 3
  rope_temporal_ratio: 1
data:
  distances: [3, 5, 7]
  noise_model: phenomenological
  rounds_policy: distance
  train_sweep: {start: 0.002, stop: 0.017, count: 20, spacing: geomspace}
  eval_sweep: {start: 0.0015, stop: 0.01, count: 10, spacing: geomspace}
  total_train_samples: 6000000
  validation_fraction: 0.05
training:
  batch_size: 1024
  num_epochs: 13
  peak_lr: 0.0001
  warmup_steps: 2000
  focal_gamma: 2.0
  focal_alpha: 0.75
  seed: 42
evaluation:
  num_test: 100000
  reference_csv: results/evaluation_results.csv
  threshold_pairs: [[3, 5], [5, 7]]
paths:
  result_dir: results
  baseline_dir: results/baseline
  run_dir: results/runs
```

`configs/laptop/d3-smoke.yaml`

```yaml
experiment_name: d3-smoke
model:
  d_model: 128
  num_heads: 4
  num_layers_by_distance: {3: 4}
  ffn_dim_by_distance: {3: 1024}
  pos_encoding: rope
  rope_spatial_ratio: 3
  rope_temporal_ratio: 1
data:
  distances: [3]
  noise_model: phenomenological
  rounds_policy: distance
  train_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  eval_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  total_train_samples: 4096
  validation_fraction: 0.125
training:
  batch_size: 64
  num_epochs: 1
  peak_lr: 0.0001
  warmup_steps: 8
  focal_gamma: 2.0
  focal_alpha: 0.75
  seed: 7
evaluation:
  num_test: 2048
  reference_csv: results/evaluation_results.csv
  threshold_pairs: [[3, 3]]
paths:
  result_dir: results
  baseline_dir: results/baseline
  run_dir: results/runs
```

- [ ] **Step 4: Rerun the config tests**

Run: `uv run pytest tests/unit/test_config_io.py -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/config/__init__.py src/transformerqec/config/schema.py src/transformerqec/config/io.py configs/baseline/current.yaml configs/laptop/d3-smoke.yaml tests/unit/test_config_io.py
git commit -m "feat: add checked-in baseline and smoke configs"
```

### Task 4: Extract the rotated surface-code and dataset helpers

**Files:**
- Create: `src/transformerqec/codes/__init__.py`
- Create: `src/transformerqec/codes/surface_code.py`
- Create: `src/transformerqec/data/__init__.py`
- Create: `src/transformerqec/data/sampling.py`
- Create: `tests/unit/test_surface_code.py`
- Create: `tests/integration/test_dataset_sampling.py`

- [ ] **Step 1: Write failing circuit and sampling tests**

```python
import numpy as np

from transformerqec.codes.surface_code import extract_detector_coordinates, make_rotated_memory_z_circuit
from transformerqec.data.sampling import generate_dataset, sample_syndromes


def test_d3_detector_count_matches_current_checkpoint() -> None:
    circuit = make_rotated_memory_z_circuit(distance=3, physical_error_rate=0.01)
    assert circuit.num_detectors == 24


def test_detector_coordinates_are_normalized() -> None:
    coords = extract_detector_coordinates(distance=5)
    assert coords.shape == (120, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords <= 1.0)


def test_generate_dataset_shapes_are_consistent() -> None:
    batch = generate_dataset(distance=3, p_values=[0.005, 0.01], shots_per_p=32)
    assert batch.syndromes.shape == (64, 24)
    assert batch.labels.shape == (64,)
    assert batch.physical_error_rates.shape == (64,)
    assert batch.coords.shape == (24, 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_surface_code.py tests/integration/test_dataset_sampling.py -q`

Expected: FAIL with `ImportError` for `transformerqec.codes.surface_code`

- [ ] **Step 3: Implement the circuit and data modules**

`src/transformerqec/codes/surface_code.py`

```python
import numpy as np
import stim


def make_rotated_memory_z_circuit(
    distance: int,
    physical_error_rate: float,
    rounds: int | None = None,
    noise_model: str = "phenomenological",
) -> stim.Circuit:
    if rounds is None:
        rounds = distance
    kwargs = {"distance": distance, "rounds": rounds}
    if noise_model == "code_capacity":
        kwargs.update({"rounds": 1, "before_round_data_depolarization": physical_error_rate})
    else:
        kwargs.update(
            {
                "before_round_data_depolarization": physical_error_rate,
                "before_measure_flip_probability": physical_error_rate,
            }
        )
    return stim.Circuit.generated("surface_code:rotated_memory_z", **kwargs)


def extract_detector_coordinates(distance: int, rounds: int | None = None) -> np.ndarray:
    circuit = make_rotated_memory_z_circuit(distance=distance, physical_error_rate=0.01, rounds=rounds)
    raw = circuit.get_detector_coordinates()
    coords = np.zeros((circuit.num_detectors, 3), dtype=np.float32)
    for detector_index, coord_tuple in raw.items():
        coords[detector_index] = coord_tuple[:3]
    for axis in range(3):
        low, high = coords[:, axis].min(), coords[:, axis].max()
        if high > low:
            coords[:, axis] = (coords[:, axis] - low) / (high - low)
    return coords
```

`src/transformerqec/data/sampling.py`

```python
from dataclasses import dataclass

import numpy as np

from transformerqec.codes.surface_code import extract_detector_coordinates, make_rotated_memory_z_circuit


@dataclass(frozen=True)
class DatasetBatch:
    syndromes: np.ndarray
    labels: np.ndarray
    physical_error_rates: np.ndarray
    coords: np.ndarray


def sample_syndromes(circuit, num_shots: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler()
    syndromes, observables = sampler.sample(num_shots, separate_observables=True)
    return syndromes.astype(np.float32), observables[:, 0].astype(np.int64)


def generate_dataset(distance: int, p_values: list[float], shots_per_p: int) -> DatasetBatch:
    syndromes, labels, rates = [], [], []
    for p in p_values:
        sampled_syndromes, sampled_labels = sample_syndromes(
            make_rotated_memory_z_circuit(distance=distance, physical_error_rate=p),
            shots_per_p,
        )
        syndromes.append(sampled_syndromes)
        labels.append(sampled_labels)
        rates.append(np.full(shots_per_p, p, dtype=np.float32))
    return DatasetBatch(
        syndromes=np.concatenate(syndromes),
        labels=np.concatenate(labels),
        physical_error_rates=np.concatenate(rates),
        coords=extract_detector_coordinates(distance=distance),
    )
```

- [ ] **Step 4: Rerun the extraction tests**

Run: `uv run pytest tests/unit/test_surface_code.py tests/integration/test_dataset_sampling.py -q`

Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/codes/__init__.py src/transformerqec/codes/surface_code.py src/transformerqec/data/__init__.py src/transformerqec/data/sampling.py tests/unit/test_surface_code.py tests/integration/test_dataset_sampling.py
git commit -m "feat: extract rotated surface-code sampling utilities"
```

### Task 5: Extract configurable RoPE utilities

**Files:**
- Create: `src/transformerqec/models/__init__.py`
- Create: `src/transformerqec/models/rope.py`
- Create: `tests/unit/test_rope.py`

- [ ] **Step 1: Write failing RoPE tests**

```python
import jax.numpy as jnp

from transformerqec.models.rope import apply_rope, build_rope_2_5d, split_rope_dimensions


def test_rope_dimension_split_preserves_head_dim() -> None:
    spatial_dim, temporal_dim = split_rope_dimensions(head_dim=128, spatial_ratio=3, temporal_ratio=1)
    assert spatial_dim == 96
    assert temporal_dim == 32


def test_rope_tables_match_sequence_length() -> None:
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    rope_cos, rope_sin = build_rope_2_5d(coords=coords, head_dim=32, seq_len=24, spatial_ratio=3, temporal_ratio=1)
    assert rope_cos.shape == (24, 16)
    assert rope_sin.shape == (24, 16)


def test_apply_rope_preserves_tensor_shape() -> None:
    x = jnp.ones((2, 4, 24, 32), dtype=jnp.float32)
    rope_cos = jnp.ones((24, 16), dtype=jnp.float32)
    rope_sin = jnp.zeros((24, 16), dtype=jnp.float32)
    rotated = apply_rope(x, rope_cos, rope_sin)
    assert rotated.shape == x.shape
```

- [ ] **Step 2: Run the RoPE tests to verify they fail**

Run: `uv run pytest tests/unit/test_rope.py -q`

Expected: FAIL with `ImportError` for `transformerqec.models.rope`

- [ ] **Step 3: Implement the reusable RoPE helpers**

`src/transformerqec/models/rope.py`

```python
import jax.numpy as jnp


def _round_even(value: float) -> int:
    return int(2 * round(value / 2))


def split_rope_dimensions(head_dim: int, spatial_ratio: int, temporal_ratio: int) -> tuple[int, int]:
    total = spatial_ratio + temporal_ratio
    spatial_dim = _round_even(head_dim * spatial_ratio / total)
    spatial_dim = max(2, min(spatial_dim, head_dim - 2))
    temporal_dim = head_dim - spatial_dim
    return spatial_dim, temporal_dim


def build_rope_2_5d(
    coords: jnp.ndarray,
    head_dim: int,
    seq_len: int,
    spatial_ratio: int,
    temporal_ratio: int,
    base_spatial: float = 10000.0,
    base_temporal: float = 10000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    spatial_dim, temporal_dim = split_rope_dimensions(head_dim, spatial_ratio, temporal_ratio)
    x_pos = coords[:, 0] * seq_len
    y_pos = coords[:, 1] * seq_len
    t_pos = coords[:, 2] * seq_len
    spatial_pairs = spatial_dim // 2
    temporal_pairs = temporal_dim // 2
    x_pairs = spatial_pairs // 2
    y_pairs = spatial_pairs - x_pairs
    freq_x = 1.0 / (base_spatial ** (2.0 * jnp.arange(x_pairs) / spatial_dim))
    freq_y = 1.0 / (base_spatial ** (2.0 * jnp.arange(y_pairs) / spatial_dim))
    freq_t = 1.0 / (base_temporal ** (2.0 * jnp.arange(temporal_pairs) / temporal_dim))
    angles_x = x_pos[:, None] * freq_x[None, :]
    angles_y = y_pos[:, None] * freq_y[None, :]
    angles_t = t_pos[:, None] * freq_t[None, :]
    spatial_angles = jnp.concatenate([angles_x, angles_y], axis=-1)
    all_angles = jnp.concatenate([spatial_angles, angles_t], axis=-1)
    return jnp.cos(all_angles), jnp.sin(all_angles)


def apply_rope(x: jnp.ndarray, rope_cos: jnp.ndarray, rope_sin: jnp.ndarray) -> jnp.ndarray:
    first_half, second_half = jnp.split(x, 2, axis=-1)
    rope_cos = rope_cos[None, None, :, :]
    rope_sin = rope_sin[None, None, :, :]
    rotated_first = first_half * rope_cos - second_half * rope_sin
    rotated_second = first_half * rope_sin + second_half * rope_cos
    return jnp.concatenate([rotated_first, rotated_second], axis=-1)
```

- [ ] **Step 4: Rerun the RoPE tests**

Run: `uv run pytest tests/unit/test_rope.py -q`

Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/models/__init__.py src/transformerqec/models/rope.py tests/unit/test_rope.py
git commit -m "feat: extract configurable 2.5d rope utilities"
```

### Task 6: Port the Flax model in a checkpoint-compatible shape

**Files:**
- Create: `src/transformerqec/models/transformer.py`
- Create: `tests/unit/test_transformer_model.py`
- Create: `tests/regression/test_legacy_checkpoint_forward.py`

- [ ] **Step 1: Write failing model tests**

```python
import jax
import jax.numpy as jnp
from pathlib import Path

from transformerqec.artifacts.io import load_checkpoint_bundle
from transformerqec.models.transformer import TransformerQEC, build_model_for_distance


def test_transformer_forward_shape() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=4, ffn_dim=1024)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, 24), dtype=jnp.float32),
        jnp.array([0.005, 0.01], dtype=jnp.float32),
        coords,
    )
    logits = model.apply(params, jnp.zeros((2, 24), dtype=jnp.float32), jnp.array([0.005, 0.01]), coords)
    assert logits.shape == (2, 2)


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
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run: `uv run pytest tests/unit/test_transformer_model.py tests/regression/test_legacy_checkpoint_forward.py -q`

Expected: FAIL with `ImportError` for `transformerqec.models.transformer`

- [ ] **Step 3: Port the model with minimal structural drift from the notebook**

`src/transformerqec/models/transformer.py`

```python
import flax.linen as nn
import jax.numpy as jnp

from transformerqec.models.rope import apply_rope, build_rope_2_5d


class TransformerBlockWithRoPE(nn.Module):
    d_model: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, rope_cos: jnp.ndarray, rope_sin: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nn.LayerNorm()(x)
        qkv = nn.Dense(self.d_model * 3)(y)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        batch_size, seq_len, _ = q.shape
        head_dim = self.d_model // self.num_heads
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        attn = nn.softmax(jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.sqrt(head_dim), axis=-1)
        mixed = jnp.einsum("bhij,bhjd->bhid", attn, v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        x = residual + nn.Dense(self.d_model)(mixed)
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.ffn_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        return x + y


class TransformerQEC(nn.Module):
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    rope_spatial_ratio: int = 3
    rope_temporal_ratio: int = 1

    @nn.compact
    def __call__(self, syndromes: jnp.ndarray, physical_error_rates: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.d_model)(syndromes[..., None])
        p_embed = nn.Dense(self.d_model)(physical_error_rates[:, None])
        x = x + p_embed[:, None, :]
        rope_cos, rope_sin = build_rope_2_5d(
            coords=coords,
            head_dim=self.d_model // self.num_heads,
            seq_len=coords.shape[0],
            spatial_ratio=self.rope_spatial_ratio,
            temporal_ratio=self.rope_temporal_ratio,
        )
        for _ in range(self.num_layers):
            x = TransformerBlockWithRoPE(self.d_model, self.num_heads, self.ffn_dim)(x, rope_cos, rope_sin)
        pooled = nn.LayerNorm()(x.mean(axis=1))
        pooled = nn.Dense(self.d_model)(pooled)
        pooled = nn.gelu(pooled)
        return nn.Dense(2)(pooled)


def build_model_for_distance(config: dict) -> TransformerQEC:
    return TransformerQEC(
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        ffn_dim=int(config["ffn_dim"]),
    )
```

- [ ] **Step 4: Rerun the model tests**

Run: `uv run pytest tests/unit/test_transformer_model.py tests/regression/test_legacy_checkpoint_forward.py -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/models/transformer.py tests/unit/test_transformer_model.py tests/regression/test_legacy_checkpoint_forward.py
git commit -m "feat: port transformer model into package"
```

### Task 7: Extract the loss, optimizer, and training loop

**Files:**
- Create: `src/transformerqec/training/__init__.py`
- Create: `src/transformerqec/training/losses.py`
- Create: `src/transformerqec/training/state.py`
- Create: `src/transformerqec/training/loop.py`
- Create: `tests/unit/test_focal_loss.py`
- Create: `tests/integration/test_training_smoke.py`

- [ ] **Step 1: Write failing training tests**

```python
import jax.numpy as jnp

from transformerqec.training.losses import focal_loss
from transformerqec.training.loop import train_step
from transformerqec.training.state import create_train_state
from transformerqec.models.transformer import TransformerQEC
import jax


def test_focal_loss_is_small_for_easy_correct_predictions() -> None:
    logits = jnp.array([[6.0, -6.0], [-6.0, 6.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss = focal_loss(logits, labels, gamma=2.0, alpha=0.75)
    assert float(loss) < 0.01


def test_train_step_runs_on_tiny_batch() -> None:
    model = TransformerQEC(d_model=128, num_heads=4, num_layers=1, ffn_dim=128)
    coords = jnp.zeros((24, 3), dtype=jnp.float32)
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((4, 24), dtype=jnp.float32),
        jnp.full((4,), 0.005, dtype=jnp.float32),
        coords,
    )
    state = create_train_state(
        params=variables["params"],
        apply_fn=model.apply,
        peak_lr=1e-4,
        warmup_steps=1,
        num_steps=4,
    )
    next_state, loss = train_step(
        state,
        jnp.zeros((4, 24), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.full((4,), 0.005, dtype=jnp.float32),
        coords,
        gamma=2.0,
        alpha=0.75,
    )
    assert next_state.step == 1
    assert float(loss) >= 0.0
```

- [ ] **Step 2: Run the training tests to verify they fail**

Run: `uv run pytest tests/unit/test_focal_loss.py tests/integration/test_training_smoke.py -q`

Expected: FAIL with `ImportError` for `transformerqec.training.losses`

- [ ] **Step 3: Implement the loss and one-epoch smoke loop**

`src/transformerqec/training/losses.py`

```python
import jax
import jax.numpy as jnp


def focal_loss(logits: jnp.ndarray, labels: jnp.ndarray, gamma: float, alpha: float) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    one_hot = jax.nn.one_hot(labels, num_classes=2)
    p_t = jnp.sum(probs * one_hot, axis=-1)
    alpha_t = alpha * labels + (1.0 - alpha) * (1 - labels)
    return jnp.mean(-alpha_t * ((1.0 - p_t) ** gamma) * jnp.log(p_t + 1e-8))
```

`src/transformerqec/training/state.py`

```python
from flax.training.train_state import TrainState
import optax


def create_optimizer(peak_lr: float, warmup_steps: int, num_steps: int) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(num_steps, warmup_steps + 1),
        end_value=peak_lr / 10.0,
    )
    return optax.adamw(learning_rate=schedule)


def create_train_state(params, apply_fn, peak_lr: float, warmup_steps: int, num_steps: int) -> TrainState:
    return TrainState.create(apply_fn=apply_fn, params=params, tx=create_optimizer(peak_lr, warmup_steps, num_steps))
```

`src/transformerqec/training/loop.py`

```python
import jax
import jax.numpy as jnp

from transformerqec.training.losses import focal_loss


@jax.jit
def train_step(state, syndromes, labels, physical_error_rates, coords, gamma: float, alpha: float):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, syndromes, physical_error_rates, coords)
        return focal_loss(logits, labels, gamma=gamma, alpha=alpha)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

- [ ] **Step 4: Run the focused training tests**

Run: `uv run pytest tests/unit/test_focal_loss.py tests/integration/test_training_smoke.py -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/training/__init__.py src/transformerqec/training/losses.py src/transformerqec/training/state.py src/transformerqec/training/loop.py tests/unit/test_focal_loss.py tests/integration/test_training_smoke.py
git commit -m "feat: extract training loss and smoke loop"
```

### Task 8: Extract the PyMatching baseline and benchmark engine

**Files:**
- Create: `src/transformerqec/baselines/__init__.py`
- Create: `src/transformerqec/baselines/pymatching_decoder.py`
- Create: `src/transformerqec/evaluation/__init__.py`
- Create: `src/transformerqec/evaluation/metrics.py`
- Create: `src/transformerqec/evaluation/benchmark.py`
- Create: `tests/integration/test_benchmark_smoke.py`

- [ ] **Step 1: Write failing benchmark tests**

```python
from pathlib import Path

from transformerqec.evaluation.metrics import logical_error_rate
from transformerqec.evaluation.benchmark import write_benchmark_rows


def test_logical_error_rate_matches_fraction_of_mistakes() -> None:
    assert logical_error_rate([0, 1, 1, 0], [0, 1, 0, 0]) == 0.25


def test_write_benchmark_rows_creates_expected_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "evaluation_results.csv"
    write_benchmark_rows(
        csv_path,
        [
            {"d": 3, "p": 0.005, "mwpm_ler": 0.01, "transformer_ler": 0.009, "improvement_pct": 10.0},
        ],
    )
    text = csv_path.read_text()
    assert "mwpm_ler" in text
    assert "transformer_ler" in text
```

- [ ] **Step 2: Run the benchmark tests to verify they fail**

Run: `uv run pytest tests/integration/test_benchmark_smoke.py -q`

Expected: FAIL with `ImportError` for `transformerqec.evaluation.metrics`

- [ ] **Step 3: Implement PyMatching decode and the benchmark row writer**

`src/transformerqec/baselines/pymatching_decoder.py`

```python
import numpy as np
import pymatching


def decode_with_pymatching(circuit, syndromes: np.ndarray) -> np.ndarray:
    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model())
    predictions = [matching.decode(sample) for sample in syndromes]
    return np.asarray([prediction[0] for prediction in predictions], dtype=np.int64)
```

`src/transformerqec/evaluation/metrics.py`

```python
import math


def logical_error_rate(predictions, labels) -> float:
    total = len(labels)
    mistakes = sum(int(pred != label) for pred, label in zip(predictions, labels))
    return mistakes / total


def improvement_pct(mwpm_ler: float, transformer_ler: float) -> float:
    if mwpm_ler == 0.0:
        return math.nan
    return 100.0 * (mwpm_ler - transformer_ler) / mwpm_ler
```

`src/transformerqec/evaluation/benchmark.py`

```python
import csv
from pathlib import Path

from transformerqec.evaluation.metrics import improvement_pct, logical_error_rate


def write_benchmark_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["d", "p", "mwpm_ler", "transformer_ler", "improvement_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_threshold_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run the benchmark tests**

Run: `uv run pytest tests/integration/test_benchmark_smoke.py -q`

Expected: PASS with the benchmark smoke test creating a small CSV under a temporary directory

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/baselines/__init__.py src/transformerqec/baselines/pymatching_decoder.py src/transformerqec/evaluation/__init__.py src/transformerqec/evaluation/metrics.py src/transformerqec/evaluation/benchmark.py tests/integration/test_benchmark_smoke.py
git commit -m "feat: add pymatching baseline and benchmark writer"
```

### Task 9: Bless the current repository results into a tracked baseline directory

**Files:**
- Create: `src/transformerqec/artifacts/manifest.py`
- Create: `scripts/bless_baseline.py`
- Create: `results/baseline/manifest.json`
- Create: `tests/regression/test_baseline_manifest.py`

- [ ] **Step 1: Write a failing test for the blessed baseline manifest**

```python
import json
from pathlib import Path


def test_blessed_manifest_tracks_existing_files() -> None:
    manifest = json.loads(Path("results/baseline/manifest.json").read_text())
    tracked = {item["relative_path"] for item in manifest["files"]}
    assert "transformer_qec_d3.pkl" in tracked
    assert "evaluation_results.csv" in tracked
```

- [ ] **Step 2: Run the manifest test to verify it fails**

Run: `uv run pytest tests/regression/test_baseline_manifest.py -q`

Expected: FAIL with `FileNotFoundError` for `results/baseline/manifest.json`

- [ ] **Step 3: Implement the manifest helpers and bless the current artifacts**

`src/transformerqec/artifacts/manifest.py`

```python
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(output_path: Path, files: list[Path]) -> None:
    payload = {
        "baseline_name": "current",
        "files": [
            {
                "relative_path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
```

`scripts/bless_baseline.py`

```python
from pathlib import Path
import shutil

from transformerqec.artifacts.manifest import write_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results"
DEST = ROOT / "results" / "baseline"

FILES = [
    "transformer_qec_d3.pkl",
    "transformer_qec_d5.pkl",
    "transformer_qec_d7.pkl",
    "evaluation_results.csv",
    "threshold_estimates.txt",
    "logical_error_rates.png",
    "transformer_vs_mwpm.png",
]

DEST.mkdir(parents=True, exist_ok=True)
copied_files = []
for relative_name in FILES:
    source_path = SOURCE / relative_name
    dest_path = DEST / relative_name
    shutil.copy2(source_path, dest_path)
    copied_files.append(dest_path)

write_manifest(DEST / "manifest.json", copied_files)
```

- [ ] **Step 4: Run the bless script and rerun the manifest test**

Run: `uv run python scripts/bless_baseline.py && uv run pytest tests/regression/test_baseline_manifest.py -q`

Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/artifacts/manifest.py scripts/bless_baseline.py results/baseline tests/regression/test_baseline_manifest.py
git commit -m "feat: bless current repo artifacts as baseline"
```

### Task 10: Wire the CLI to the package workflows and baseline checks

**Files:**
- Modify: `src/transformerqec/cli.py`
- Create: `tests/smoke/test_cli_smoke.py`

- [ ] **Step 1: Write failing CLI tests**

```python
from pathlib import Path

from typer.testing import CliRunner

from transformerqec.cli import app

runner = CliRunner()


def test_help_mentions_reproduce_baseline() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "reproduce-baseline" in result.stdout


def test_reproduce_baseline_accepts_checked_in_config() -> None:
    result = runner.invoke(app, ["reproduce-baseline", "--config", str(Path("configs/laptop/d3-smoke.yaml"))])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run: `uv run pytest tests/smoke/test_cli_smoke.py -q`

Expected: FAIL because the CLI has no commands yet

- [ ] **Step 3: Implement the first stable CLI surface**

`src/transformerqec/cli.py`

```python
from pathlib import Path

import typer

from transformerqec.config.io import load_run_config

app = typer.Typer(name="transformerqec", help="TransformerQEC decoder library CLI")


@app.command("generate")
def generate(config: Path) -> None:
    load_run_config(config)
    typer.echo(f"Loaded data generation config: {config}")


@app.command("train")
def train(config: Path) -> None:
    load_run_config(config)
    typer.echo(f"Loaded training config: {config}")


@app.command("eval")
def evaluate(config: Path) -> None:
    load_run_config(config)
    typer.echo(f"Loaded evaluation config: {config}")


@app.command("reproduce-baseline")
def reproduce_baseline(config: Path) -> None:
    loaded = load_run_config(config)
    typer.echo(f"Baseline reproduction ready for {loaded.experiment_name}")


@app.command("benchmark")
def benchmark(config: Path) -> None:
    load_run_config(config)
    typer.echo(f"Benchmark config accepted: {config}")


@app.command("infer")
def infer(config: Path) -> None:
    load_run_config(config)
    typer.echo(f"Inference config accepted: {config}")
```

- [ ] **Step 4: Rerun the CLI tests**

Run: `uv run pytest tests/smoke/test_cli_smoke.py -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/cli.py tests/smoke/test_cli_smoke.py
git commit -m "feat: add stable cli entrypoints"
```

### Task 11: Archive the notebooks and publish the package-facing docs

**Files:**
- Modify: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/baseline-reproduction.md`
- Create: `docs/research-landscape.md`
- Create: `docs/research-benchmark-contract.md`
- Create: `notebooks/README.md`
- Create: `tests/smoke/test_docs_contract.py`

- [ ] **Step 1: Write failing docs contract tests**

```python
from pathlib import Path


def test_required_docs_exist() -> None:
    assert Path("docs/architecture.md").exists()
    assert Path("docs/baseline-reproduction.md").exists()
    assert Path("docs/research-landscape.md").exists()
    assert Path("docs/research-benchmark-contract.md").exists()


def test_readme_mentions_uv_and_baseline_command() -> None:
    text = Path("README.md").read_text()
    assert "uv sync" in text
    assert "uv run transformerqec reproduce-baseline" in text
```

- [ ] **Step 2: Run the docs tests to verify they fail**

Run: `uv run pytest tests/smoke/test_docs_contract.py -q`

Expected: FAIL because the docs files do not exist yet

- [ ] **Step 3: Write the package-facing docs and archive guidance**

`README.md`

````markdown
# TransformerQEC

## Quickstart

```bash
uv sync --extra dev
uv run transformerqec reproduce-baseline --config configs/laptop/d3-smoke.yaml
```

## Validation Standard

Every baseline or research claim in this repo is expected to resolve to a checked-in config, a recorded artifact set, and a reproducible comparison against a blessed baseline.
````

`docs/architecture.md`

```markdown
# Architecture

`transformerqec.codes` owns STIM circuit construction and detector coordinates.
`transformerqec.data` owns dataset generation.
`transformerqec.models` owns the reusable Transformer and RoPE implementation.
`transformerqec.training`, `transformerqec.evaluation`, and `transformerqec.research` own the executable workflows.
```

`docs/baseline-reproduction.md`

```markdown
# Baseline Reproduction

1. Run `uv sync --extra dev`.
2. Run `uv run python scripts/bless_baseline.py`.
3. Run `uv run transformerqec reproduce-baseline --config configs/laptop/d3-smoke.yaml`.
4. Compare outputs under `results/runs/` against `results/baseline/manifest.json`.
```

`docs/research-landscape.md`

```markdown
# Research Landscape

- AlphaQubit: recurrent and transformer decoding under circuit-level noise.
- Transformer-QEC repos: attention over syndrome strings with learned positional structure.
- PyMatching and sparse blossom: strong classical graph-matching baselines.
- TransformerQEC differentiator: explicit `(2+1)D` RoPE with a checked-in regression harness and public baseline artifacts.
```

`docs/research-benchmark-contract.md`

```markdown
# Research Benchmark Contract

Every candidate method must declare:

- the exact config it uses,
- the target metric it aims to improve,
- the baseline artifact set it compares against,
- the saved outputs it produces,
- the written conclusion that accepts or rejects the change.
```

`notebooks/README.md`

```markdown
# Notebooks

The original notebooks are historical artifacts. Once package parity is in place, move them under `notebooks/archive/` and treat the package CLI plus docs as the canonical workflow.
```

- [ ] **Step 4: Archive the original notebooks and rerun the docs tests**

Run: `mkdir -p notebooks/archive && git mv notebooks/01_data_exploration.ipynb notebooks/archive/01_data_exploration.ipynb && git mv notebooks/02_model_and_training.ipynb notebooks/archive/02_model_and_training.ipynb && git mv notebooks/03_evaluation.ipynb notebooks/archive/03_evaluation.ipynb && uv run pytest tests/smoke/test_docs_contract.py -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add README.md docs/architecture.md docs/baseline-reproduction.md docs/research-landscape.md docs/research-benchmark-contract.md notebooks/README.md notebooks/archive tests/smoke/test_docs_contract.py
git commit -m "docs: publish package workflow and archive notebooks"
```

### Task 12: Add the research registry and the first reproducible ablation pair

**Files:**
- Create: `src/transformerqec/research/__init__.py`
- Create: `src/transformerqec/research/registry.py`
- Create: `src/transformerqec/research/compare.py`
- Create: `configs/experiments/rope_ratio_3_1.yaml`
- Create: `configs/experiments/rope_ratio_1_1.yaml`
- Create: `tests/integration/test_research_registry.py`

- [ ] **Step 1: Write failing research harness tests**

```python
from transformerqec.research.registry import get_candidate, list_candidates


def test_rope_ratio_candidates_are_registered() -> None:
    names = [candidate.name for candidate in list_candidates()]
    assert "rope-ratio-3-1" in names
    assert "rope-ratio-1-1" in names


def test_candidate_lookup_returns_target_metric() -> None:
    candidate = get_candidate("rope-ratio-3-1")
    assert candidate.target_metric == "transformer_ler"
```

- [ ] **Step 2: Run the research tests to verify they fail**

Run: `uv run pytest tests/integration/test_research_registry.py -q`

Expected: FAIL with `ImportError` for `transformerqec.research.registry`

- [ ] **Step 3: Implement the first research contract around a low-risk ablation**

`src/transformerqec/research/registry.py`

```python
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
    return _CANDIDATES[name]
```

`src/transformerqec/research/compare.py`

```python
from transformerqec.artifacts.io import load_evaluation_rows


def compare_csvs(reference_csv, candidate_csv):
    reference_rows = load_evaluation_rows(reference_csv)
    candidate_rows = load_evaluation_rows(candidate_csv)
    assert len(reference_rows) == len(candidate_rows)
    return [
        {
            "distance": reference["distance"],
            "physical_error_rate": reference["physical_error_rate"],
            "reference_transformer_ler": reference["transformer_ler"],
            "candidate_transformer_ler": candidate["transformer_ler"],
            "delta_transformer_ler": candidate["transformer_ler"] - reference["transformer_ler"],
        }
        for reference, candidate in zip(reference_rows, candidate_rows)
    ]
```

`configs/experiments/rope_ratio_3_1.yaml`

```yaml
experiment_name: rope-ratio-3-1
model:
  d_model: 128
  num_heads: 4
  num_layers_by_distance: {3: 4}
  ffn_dim_by_distance: {3: 1024}
  pos_encoding: rope
  rope_spatial_ratio: 3
  rope_temporal_ratio: 1
data:
  distances: [3]
  noise_model: phenomenological
  rounds_policy: distance
  train_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  eval_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  total_train_samples: 4096
  validation_fraction: 0.125
training:
  batch_size: 64
  num_epochs: 1
  peak_lr: 0.0001
  warmup_steps: 8
  focal_gamma: 2.0
  focal_alpha: 0.75
  seed: 7
evaluation:
  num_test: 2048
  reference_csv: results/baseline/evaluation_results.csv
  threshold_pairs: [[3, 3]]
paths:
  result_dir: results
  baseline_dir: results/baseline
  run_dir: results/runs
```

`configs/experiments/rope_ratio_1_1.yaml`

```yaml
experiment_name: rope-ratio-1-1
model:
  d_model: 128
  num_heads: 4
  num_layers_by_distance: {3: 4}
  ffn_dim_by_distance: {3: 1024}
  pos_encoding: rope
  rope_spatial_ratio: 1
  rope_temporal_ratio: 1
data:
  distances: [3]
  noise_model: phenomenological
  rounds_policy: distance
  train_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  eval_sweep: {start: 0.005, stop: 0.01, count: 2, spacing: geomspace}
  total_train_samples: 4096
  validation_fraction: 0.125
training:
  batch_size: 64
  num_epochs: 1
  peak_lr: 0.0001
  warmup_steps: 8
  focal_gamma: 2.0
  focal_alpha: 0.75
  seed: 7
evaluation:
  num_test: 2048
  reference_csv: results/baseline/evaluation_results.csv
  threshold_pairs: [[3, 3]]
paths:
  result_dir: results
  baseline_dir: results/baseline
  run_dir: results/runs
```

- [ ] **Step 4: Rerun the research tests**

Run: `uv run pytest tests/integration/test_research_registry.py -q`

Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/transformerqec/research/__init__.py src/transformerqec/research/registry.py src/transformerqec/research/compare.py configs/experiments/rope_ratio_3_1.yaml configs/experiments/rope_ratio_1_1.yaml tests/integration/test_research_registry.py
git commit -m "feat: add research registry and first rope ablation configs"
```

## Final Verification Matrix

Run these after the last task:

1. `uv sync --extra dev`
2. `uv run pytest tests/unit -q`
3. `uv run pytest tests/integration -q`
4. `uv run pytest tests/regression -q`
5. `uv run pytest tests/smoke -q`
6. `uv run python scripts/bless_baseline.py`
7. `uv run transformerqec reproduce-baseline --config configs/laptop/d3-smoke.yaml`
8. `uv run transformerqec benchmark --config configs/experiments/rope_ratio_3_1.yaml`

Expected end state:

- `uv` installs cleanly on a laptop,
- the package imports without notebooks,
- the current repo artifacts are preserved under `results/baseline/`,
- the docs explain the stable workflow and the research quality bar,
- the repo contains a concrete first ablation pair that can be evaluated against the blessed baseline.
