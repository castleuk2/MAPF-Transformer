from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class ModelConfig:
    """Architecture and input-layout configuration.

    The default layout is intentionally exact:
    15 frames * (16 agents + 1 transition) + 1 ACT query = 256 tokens.
    """

    map_size: int = 15
    map_latents: int = 16
    one_hop_ctg: bool = False
    max_neighbors: int = 15
    history_frames: int = 15
    d_model: int = 256
    n_heads: int = 8
    temporal_layers: int = 8
    spatial_latent_layers: int = 1
    dropout: float = 0.1
    mlp_ratio: int = 4
    num_actions: int = 5
    distance_buckets: int = 64
    cell_states: int = 2
    aux_map_reconstruction: bool = True
    aux_map_loss_weight: float = 0.05

    @property
    def agents_per_frame(self) -> int:
        return self.max_neighbors + 1

    @property
    def tokens_per_frame(self) -> int:
        return self.agents_per_frame + 1

    @property
    def context_tokens(self) -> int:
        return self.history_frames * self.tokens_per_frame + 1

    @property
    def ego_slot(self) -> int:
        return self.max_neighbors

    @property
    def local_radius(self) -> int:
        return self.map_size // 2

    def validate(self) -> None:
        if self.map_size <= 0 or self.map_size % 2 == 0:
            raise ValueError("map_size must be a positive odd number so Ego has a center cell")
        if self.map_latents <= 0:
            raise ValueError("map_latents must be positive")
        if self.max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.context_tokens != 256:
            raise ValueError(
                "The reference architecture requires exactly 256 context tokens; "
                f"received {self.context_tokens}. Adjust history_frames/max_neighbors."
            )
        if self.distance_buckets != 64:
            raise ValueError("The 6-bit distance field requires 64 buckets")
        if self.num_actions != 5:
            raise ValueError("POGEMA-compatible action space requires 5 actions")


@dataclass(slots=True)
class TrainingConfig:
    train_manifest: str = "data/train/manifest.jsonl"
    val_manifest: str | None = "data/val/manifest.jsonl"
    output_dir: str = "runs/mapf_transformer"
    batch_size: int = 16
    val_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    amp: bool = True
    log_every: int = 20
    validate_every: int = 500
    save_every: int = 1000
    history_augmentation: bool = True
    min_history_frames: int = 1
    # Keep this fraction of the already-stored final-goal WAIT suffix as
    # supervised targets. All frames remain available when constructing history.
    goal_wait_keep_ratio: float = 0.0
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    def validate(self, model: ModelConfig) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.val_batch_size is not None and self.val_batch_size <= 0:
            raise ValueError("val_batch_size must be positive when provided")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative")
        if not 1 <= self.min_history_frames <= model.history_frames:
            raise ValueError("min_history_frames must be within the model history window")
        if not 0.0 <= self.goal_wait_keep_ratio <= 1.0:
            raise ValueError("goal_wait_keep_ratio must be in [0, 1]")


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.model.validate()
        self.training.validate(self.model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _filter_dataclass_values(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return {key: value for key, value in values.items() if key in allowed}


def experiment_config_from_dict(values: Mapping[str, Any]) -> ExperimentConfig:
    model_values = _filter_dataclass_values(ModelConfig, values.get("model", {}))
    training_values = _filter_dataclass_values(TrainingConfig, values.get("training", {}))
    config = ExperimentConfig(
        model=ModelConfig(**model_values),
        training=TrainingConfig(**training_values),
    )
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return experiment_config_from_dict(values)


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False)
