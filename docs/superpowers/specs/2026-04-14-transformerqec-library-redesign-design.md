# TransformerQEC Library Redesign Design

Date: 2026-04-14
Status: Draft for review
Owner intent: Convert the current notebook-first repo into a clean reusable decoder library, managed with UV, reproducible on a personal computer, with verified baseline results to guard against regressions during refactor, and with a research-engineering standard that reads as serious to both academic readers and strong industrial research teams.

## 1. Goal

Redesign TransformerQEC from a notebook-owned experiment into a maintainable Python package that:

- runs end to end on a laptop without Colab or Google Drive,
- reproduces the current notebook baseline as a verification target,
- supports future decoder engineering improvements without forcing code back into notebooks,
- supports a post-refactor research phase where new decoder ideas are tested against the reproduced baseline,
- exposes a small stable interface for data generation, training, evaluation, and inference,
- documents prior research and adjacent projects so the repo is legible in historical context.

The redesign optimizes for a clean reusable decoder library first. Research acceleration and novelty maximization are supported, but they are downstream of baseline reproducibility and maintainability.

### 1.1 North Star

The North Star is not just to clean up the repo. It is to turn TransformerQEC into a public research-engineering artifact that looks materially stronger than the typical notebook-centric academic release and credible even to reviewers from highly disciplined research organizations.

Because "better than DeepMind's platform" is not directly verifiable from the outside, the spec should translate that ambition into public, measurable qualities:

- stronger local reproducibility,
- clearer artifact provenance,
- tighter regression and benchmark discipline,
- cleaner modular code boundaries,
- faster onboarding for an external reader,
- more rigorous experiment comparison and writeup hygiene.

In practice, the repo should aim to look like a small but serious quantitative research platform rather than a one-off experiment dump.

## 2. Scope

### In scope

- UV-based Python project setup
- extraction of notebook logic into package modules under `src/`
- command-line entrypoints for the current workflow
- reproducible baseline artifacts for regression checking
- replacement of Colab/Drive assumptions with local paths
- history/context documentation for prior papers and similar repos
- notebook retention as examples and analysis surfaces, not primary code owners
- tests for core behavior and baseline parity checks
- a documented phase-two research workflow for developing and validating new decoder variants against the reproduced baseline

### Out of scope for the first redesign pass

- inventing a substantially new decoder architecture
- changing the primary noise model away from the current phenomenological baseline
- forcing GPU support as a requirement
- building a full interactive web UI
- adding every possible decoder baseline at once
- promising exact bitwise reproducibility across every machine and backend

## 3. Success Criteria

The redesign is successful when all of the following hold:

1. A fresh user can clone the repo, install with UV, and run a laptop-safe baseline workflow without notebooks.
2. The baseline workflow reproduces the current committed results closely enough to detect behavioral regressions.
3. All core logic currently trapped in notebooks is moved into importable modules.
4. The notebooks become thin clients over package APIs or are archived as historical notebooks.
5. The repo includes a historical-context document covering similar research and public implementations.
6. The repo layout makes it straightforward to add and compare future decoder variants.
7. The spec defines how post-refactor research ideas are proposed, implemented, benchmarked, and accepted as genuine improvements over the original baseline.
8. An external reviewer can inspect the repo and see a clear research-engineering standard: reproducible commands, versioned artifacts, benchmark contracts, and documented conclusions instead of notebook-only claims.

## 4. Recommended Architecture

The redesign follows a layered decoder-library structure.

### 4.1 Package layout

```text
TransformerQEC/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   └── transformerqec/
│       ├── __init__.py
│       ├── config/
│       ├── cli/
│       ├── codes/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── baselines/
│       ├── evaluation/
│       ├── artifacts/
│       ├── research/
│       └── utils/
├── configs/
│   ├── baseline/
│   ├── laptop/
│   └── experiments/
├── tests/
│   ├── smoke/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── scripts/
├── docs/
│   ├── superpowers/specs/
│   ├── research-landscape.md
│   ├── baseline-reproduction.md
│   └── architecture.md
├── notebooks/
│   ├── examples/
│   └── archive/
├── results/
│   └── baseline/
└── references/
```

### 4.2 Module responsibilities

#### `transformerqec.codes`

- STIM circuit builders for rotated surface-code memory experiments
- code-distance and rounds helpers
- detector-coordinate extraction and normalization
- future home for multiple code families if the repo expands later

#### `transformerqec.data`

- syndrome and logical observable sampling
- dataset generation over `p` grids
- train/val/test splits
- optional caching to local files
- dataset manifests containing exact generation parameters

#### `transformerqec.models`

- current TransformerQEC model
- `(2+1)D` anisotropic RoPE implementation
- model config dataclasses / schemas
- future model variants behind explicit names

#### `transformerqec.training`

- focal loss
- optimizer configuration
- train-state creation
- epoch execution
- checkpoint selection and export

#### `transformerqec.baselines`

- PyMatching baseline decoder wrapper
- baseline config and metadata capture

#### `transformerqec.evaluation`

- logical error rate sweeps
- confidence intervals
- threshold estimation
- comparison tables
- regression metrics against blessed baselines

#### `transformerqec.artifacts`

- checkpoint schema
- plot generation
- CSV and JSON summaries
- run manifests that include parameter ranges and provenance

#### `transformerqec.cli`

Commands should support:

- `transformerqec generate`
- `transformerqec train`
- `transformerqec eval`
- `transformerqec benchmark`
- `transformerqec reproduce-baseline`
- `transformerqec infer`

#### `transformerqec.research`

This namespace is not required to be fully populated in the first implementation, but the design must reserve it for:

- ablation registries
- comparison harness glue
- candidate novel-engineering variants
- experiment manifests for post-baseline research runs
- result ranking against the blessed baseline

The point is to keep research code additive instead of contaminating the stable baseline modules.

## 5. Baseline Reproduction and Regression Protection

This is a first-class requirement, not a cleanup afterthought.

### 5.1 Baseline contract

The refactor must preserve a known baseline artifact set derived from the current notebook pipeline or from a newly reblessed canonical rerun.

The repo should define one explicit baseline profile, for example:

- code distances included
- physical error-rate train grid
- physical error-rate eval grid
- rounds policy
- model hyperparameters
- sample counts
- checkpoint filenames

This baseline profile must be stored in versioned config and referenced by tests/docs.

### 5.2 Blessed artifacts

The repo should maintain a `results/baseline/` directory or equivalent that includes:

- blessed checkpoints
- blessed evaluation CSV
- blessed threshold summary
- blessed plot images
- blessed manifest describing exactly how these were produced

### 5.3 Regression checks

Regression checks should verify:

- checkpoint loadability
- detector coordinate extraction shape and normalization
- training config parity with baseline profile
- evaluation pipeline output schema
- numeric closeness to the blessed baseline

For numeric regression, exact bitwise equality is not required unless achieved naturally. Instead, define explicit tolerances, such as:

- exact equality for config fields and artifact schemas
- exact equality for eval grid values
- bounded absolute/relative differences for logical error rates
- bounded slope differences for derived regression summaries

### 5.4 Local verification tiers

#### Tier 1: smoke

Runs on any laptop quickly:

- tiny STIM sample generation
- one forward pass
- one PyMatching decode
- one minimal eval loop

#### Tier 2: baseline parity

Runs a reduced but meaningful subset of the canonical baseline and compares outputs to stored references.

#### Tier 3: full reproduction

Rebuilds the full baseline artifact set locally if the machine budget allows it. This is not required for every test run, but it must be documented and runnable.

## 6. Results and Provenance Model

Every serious run should write a manifest. The manifest should include:

- git commit
- package version
- config path
- hardware/backend info
- dependency versions
- train/eval p-ranges
- code distances
- number of shots/samples
- timestamp
- generated artifact paths

This prevents a repeat of the current mismatch between notebooks, CSVs, threshold files, and README prose.

### 6.1 Research-engineering quality bar

The redesign should explicitly optimize for public credibility.

That means the repository should present evidence in a way that would look competent to both:

- an academic reader trying to reproduce or extend the work,
- a research engineer evaluating whether the project is run with real quantitative discipline.

Concrete signs of that quality bar include:

- one-command or few-command baseline reproduction,
- machine-readable manifests for every serious run,
- canonical benchmark configs checked into the repo,
- saved comparison tables between baseline and candidate methods,
- explicit acceptance or rejection summaries for each experiment,
- documentation that explains what is stable, what is exploratory, and what is historical.

The goal is to make the repo look like a lightweight research platform with accountable experiment management, not just a cleaned-up notebook collection.

## 7. Post-Refactor Research and Development Phase

After the library reaches baseline parity, the repo should explicitly enter a second phase: improving on the original repository author's decoder through verified engineering experiments.

This phase is part of the design, not an unstructured future wish list.

### Objectives

- improve logical error rate relative to the reproduced baseline,
- improve efficiency, latency, or resource usage where possible,
- narrow the novelty gap versus existing transformer/QEC work by testing differentiated engineering ideas,
- raise the public research-engineering standard of the repo so improvements are credible, inspectable, and easy to compare,
- reject ideas that do not beat the baseline under the agreed validation framework.

### Research operating rule

No proposed "improvement" counts unless it is evaluated through the same artifact and regression framework used for the baseline.

Each new approach must have:

- a named config or experiment manifest,
- a clear hypothesis,
- a baseline comparator,
- saved metrics and artifacts,
- a written conclusion stating whether it improved, regressed, or was inconclusive,
- a comparison record that an external reader can follow without reopening a notebook.

### Candidate research directions

These are the first-class follow-on work items the refactor should make easy:

1. positional encoding variants:
   - current anisotropic `(2+1)D` RoPE
   - isotropic `x/y/t` RoPE
   - spatial-only RoPE
   - learned additive coordinate embeddings
   - ratio sweeps such as `1:1`, `2:1`, `3:1`, `4:1`
2. architecture changes:
   - deeper or wider transformer variants
   - local-global hybrid attention
   - sparse or windowed attention for longer detector sequences
   - lightweight pooling or hierarchical detector grouping
3. task formulation changes:
   - binary logical-Z prediction vs richer logical-label formulations
   - auxiliary losses
   - multi-task decoding targets
4. efficiency changes:
   - reduced precision policies
   - improved batching and data pipelines
   - faster evaluation and benchmarking paths
5. noise-model expansion:
   - circuit-level noise once the phenomenological baseline is stable

### Improvement criteria

The repo should define explicit dimensions for "better than baseline", such as:

- lower transformer logical error rate,
- better performance relative to MWPM in target regimes,
- lower latency at similar quality,
- lower memory use,
- better scaling to higher code distances,
- better robustness under out-of-distribution `p` values,
- better experiment clarity and reproducibility than the original notebook workflow.

Not every experiment must improve every axis, but each experiment must declare its target axis before running.

### Acceptance criteria for a new method

A candidate method is promoted from `research/` toward the stable library only when:

- it reproduces cleanly on a laptop-safe reduced benchmark,
- it beats the blessed baseline on at least one declared target axis,
- it does not silently change baseline evaluation semantics,
- its artifacts are versioned and comparable,
- its README/docs summary explains what changed and why the result matters,
- its benchmark evidence is organized well enough that a skeptical external reviewer can audit the claim quickly.

## 8. Notebook Strategy

The notebooks should no longer own business logic.

### Keep

- exploratory analysis notebook
- explanation notebook for detector layouts and model intuition
- result-visualization notebook that imports package APIs

### Change

- notebook cells should call package functions
- no duplicated model or evaluation code
- no Drive mounts or hardcoded Colab paths in the canonical workflow

### Archive

- preserve current notebooks in an `archive/` or clearly marked historical state if needed for provenance

## 9. Historical Context Documentation

Add a repo-level document, for example `docs/research-landscape.md`, that explains:

- the key prior papers
- what problem each prior approach addresses
- what architectures they use
- whether they target surface code, toric code, color code, or other codes
- whether they optimize for threshold, accuracy, transferability, or real-time decoding
- how TransformerQEC differs

Minimum topics to include:

- AlphaQubit
- Transformer-QEC
- recent global-receptive-field / transformer surface-code papers
- vision-transformer / mixture-of-experts QEC work
- older deep-learning decoders
- MWPM / PyMatching / sparse blossom as baseline families

This document should explicitly separate:

- what is already known in the literature
- what this repo contributes
- where the current novelty claim is narrow
- what future research directions in this repo are still plausibly differentiated

## 10. Novel Engineering Opportunities After the Refactor

The redesign should make these future directions easy, even if they are not all built immediately:

1. positional-encoding ablation framework
2. decoder-interface abstraction for side-by-side comparisons
3. artifact-verified benchmark sweeps across distances and p-ranges
4. latency benchmarking on CPU vs optional accelerator backends
5. future support for circuit-level noise
6. richer decoder outputs beyond binary logical-Z classification

The important design constraint is that future novelty experiments should be added as new modules/configs, not by rewriting baseline code paths.

These opportunities should be treated as the first wave of the post-refactor R&D phase described in Section 7.

## 11. Package Management and Environment Design

Use UV for environment management.

Requirements:

- `pyproject.toml` defines the package and dependency groups
- `uv.lock` is committed
- dependency groups include at least:
  - runtime
  - dev
  - notebook
  - optional accelerator extras if needed

The default install path should be CPU-safe.

The project must not require Colab-specific behavior to function.

## 12. CLI and User Experience

The first stable CLI should make the common path obvious:

```bash
uv sync
uv run transformerqec generate --config configs/laptop/baseline-d3.yaml
uv run transformerqec train --config configs/laptop/baseline-d3.yaml
uv run transformerqec eval --config configs/laptop/baseline-d3.yaml
uv run transformerqec reproduce-baseline --config configs/baseline/current.yaml
```

The README should have:

- 5-minute quickstart
- architecture overview
- baseline reproduction section
- artifact layout explanation
- research-landscape doc link
- a short statement of the repo's validation and experiment-quality standard

## 13. Testing Strategy

### Unit tests

- RoPE table shapes and invariants
- detector-coordinate normalization
- config parsing
- checkpoint schema round-trip

### Integration tests

- STIM generation to model input pipeline
- training one tiny epoch
- PyMatching baseline execution
- end-to-end eval on a tiny config

### Regression tests

- compare generated eval grids against blessed manifests
- compare key LER values within tolerance
- verify all baseline artifact schemas

### Documentation tests

- CLI examples in README should be runnable or trivially validated

## 14. Migration Plan

Suggested migration sequence:

1. create UV project skeleton
2. extract pure utilities from notebooks
3. extract model code
4. extract data generation and evaluation code
5. add checkpoint/artifact schemas
6. add CLI wrappers
7. build smoke tests
8. bless a canonical baseline artifact set
9. update notebooks to import package code
10. add historical-context docs
11. define the phase-two research benchmark contract
12. implement the first post-baseline ablation or improvement experiment

This sequence preserves forward motion while reducing the risk of breaking the current experimental path before baseline parity exists.

## 15. Risks and Mitigations

### Risk: refactor changes model behavior silently

Mitigation:

- baseline artifacts plus regression thresholds
- staged extraction with parity checks after each step

### Risk: current notebook artifacts are already inconsistent

Mitigation:

- bless a single canonical baseline and document that older artifacts are historical

### Risk: local laptops cannot reproduce the largest runs

Mitigation:

- define smoke, reduced baseline, and full baseline tiers
- make CPU-safe configs explicit

### Risk: research code starts polluting the stable library again

Mitigation:

- strict module boundaries
- stable CLI paths for baseline workflows
- `research/` namespace for experiments

## 16. Recommendation

Proceed with the layered decoder-library redesign.

This gives the repo:

- maintainable code boundaries,
- UV-managed reproducibility,
- a clean path to preserve current results,
- and a structured path to test future engineering ideas without hiding logic in notebooks.

The redesign should be judged first on baseline parity and maintainability, then on how effectively it supports a verified R&D loop for surpassing the original repository results.
