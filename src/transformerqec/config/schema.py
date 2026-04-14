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
