from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    # Structured-map contract.
    local_map_size: int = 17
    core_map_size: int = 15
    patch_size: int = 3
    map_tokens: int = 25

    # Revised semantic-token contract.
    max_current_agents: int = 14
    current_tokens_per_agent: int = 8
    history_tracks: int = 7
    history_steps: int = 4
    history_tokens_per_step: int = 4
    message_neighbors: int = 6
    message_query_tokens: int = 1
    num_actions: int = 5

    # Transformer dimensions.
    d_model: int = 256
    n_heads: int = 8
    map_layers: int = 1
    # Native mapf-structured-map-transformer checkpoint.  When set, it is
    # loaded strictly and its own saved architecture config is authoritative.
    map_checkpoint: str | None = None
    freeze_map_encoder: bool = True
    transformer_layers: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Spatial fusion.
    map_attention_radius: int = 1
    use_spatial_map_fusion: bool = True
    use_structure_bias: bool = True

    # Discrete ranges.
    current_coord_clip: int = 7
    history_coord_clip: int = 31
    goal_delta_clip: int = 63
    max_hops: int = 1023

    # Optional auxiliary heads.
    enable_map_reconstruction: bool = True
    enable_semantic_reconstruction: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def patches_per_side(self) -> int:
        return self.core_map_size // self.patch_size

    @property
    def current_tokens(self) -> int:
        return self.max_current_agents * self.current_tokens_per_agent

    @property
    def history_tokens(self) -> int:
        return self.history_tracks * self.history_steps * self.history_tokens_per_step

    @property
    def coordination_tokens(self) -> int:
        return self.message_query_tokens + self.message_neighbors

    @property
    def total_tokens(self) -> int:
        return self.map_tokens + self.current_tokens + self.history_tokens + self.coordination_tokens

    @property
    def current_offset(self) -> int:
        return self.map_tokens

    @property
    def history_offset(self) -> int:
        return self.map_tokens + self.current_tokens

    @property
    def coordination_offset(self) -> int:
        return self.map_tokens + self.current_tokens + self.history_tokens

    @property
    def current_coord_vocab(self) -> int:
        return 2 * self.current_coord_clip + 2  # values + PAD

    @property
    def history_coord_vocab(self) -> int:
        return 2 * self.history_coord_clip + 2

    @property
    def goal_delta_vocab(self) -> int:
        return 2 * self.goal_delta_clip + 2

    @property
    def hops_vocab(self) -> int:
        # 0..max_hops, UNREACHABLE, OVERFLOW/PAD
        return self.max_hops + 3

    def validate(self) -> None:
        if self.local_map_size != self.core_map_size + 2:
            raise ValueError("local_map_size must be core_map_size + a one-cell halo")
        if self.core_map_size % self.patch_size != 0:
            raise ValueError("core_map_size must be divisible by patch_size")
        if self.patches_per_side**2 != self.map_tokens:
            raise ValueError("map_tokens must equal patches_per_side squared")
        if self.num_actions != 5:
            raise ValueError("action order is fixed to WAIT, UP, DOWN, LEFT, RIGHT")
        if self.current_tokens_per_agent != 3 + self.num_actions:
            raise ValueError("current agent block must be P, G, R plus five candidates")
        if self.history_tokens_per_step != 4:
            raise ValueError("history block must be P, G, R, Action-Outcome")
        if self.message_query_tokens != 1:
            raise ValueError("this implementation uses one self-message query")
        if self.message_neighbors != self.history_tracks - 1:
            raise ValueError("message neighbors must match non-ego history tracks")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.total_tokens != 256:
            raise ValueError(f"token layout must be exactly 256, got {self.total_tokens}")
        if self.map_attention_radius < 0:
            raise ValueError("map_attention_radius must be non-negative")


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    val_batch_size: int = 128
    epochs: int = 10
    max_steps: int | None = None
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    amp: bool = True
    output_dir: str = "runs/base"

    # Every training item is one agent's own ego-centered view.
    ego_action_weight: float = 1.0
    ranking_loss_weight: float = 0.0
    map_loss_weight: float = 0.02
    semantic_reconstruction_weight: float = 0.05

    # 0 = no communication; >0 invokes the grouped two-pass/multi-round wrapper.
    communication_rounds: int = 0


@dataclass(slots=True)
class DataConfig:
    kind: str = "synthetic"  # synthetic | npz_manifest
    train_samples: int = 4096
    val_samples: int = 512
    train_manifest: str | None = None
    val_manifest: str | None = None
    samples_per_epoch: int = 20000
    goal_wait_keep_ratio: float = 0.2
    max_train_samples: int | None = None
    max_val_samples: int | None = 65536
    grouped_train_frames_per_epoch: int = 20000
    grouped_val_frames: int = 600
    coordinate_order: str = "row_col"  # row_col | xy
    task_mode: str = "classic"
    feature_backend: str = "python"  # python | cpp


@dataclass(slots=True)
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    values = values or {}
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {"model", "training", "data"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown top-level config sections: {unknown}")
    return ProjectConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
    )
