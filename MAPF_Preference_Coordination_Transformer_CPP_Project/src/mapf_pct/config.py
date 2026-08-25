from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    # Fixed MAPF tokenization contract.
    local_map_size: int = 17
    core_map_size: int = 15
    patch_size: int = 3
    map_tokens: int = 25
    max_agents: int = 25
    history_steps: int = 5
    history_tokens_per_agent: int = 2
    num_actions: int = 5
    relation_tokens: int = 24
    event_tokens: int = 5
    scene_tokens: int = 1
    act_tokens: int = 1

    # Transformer dimensions.
    d_model: int = 256
    n_heads: int = 8
    map_layers: int = 1
    coordination_layers: int = 4
    history_layers: int = 1
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Discrete feature vocabularies.
    goal_delta_clip: int = 31
    max_hops: int = 1023
    contender_buckets: int = 8
    outcome_states: int = 7
    delta_ctg_states: int = 6
    reason_classes: int = 12
    conflict_classes: int = 4
    scene_risk_classes: int = 3

    # Continuous feature widths.
    event_numeric_dim: int = 8
    scene_numeric_dim: int = 16
    relation_numeric_dim: int = 12

    # Optional auxiliary branches.
    enable_map_reconstruction: bool = True
    enable_reason_head: bool = True
    enable_conflict_head: bool = True
    enable_scene_risk_head: bool = True

    # Frozen Structured-25 map backbone.
    map_checkpoint: str | None = None
    freeze_map_encoder: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def patches_per_side(self) -> int:
        return self.core_map_size // self.patch_size

    @property
    def agent_tokens_per_agent(self) -> int:
        return 1 + self.history_tokens_per_agent + self.num_actions

    @property
    def agent_tokens(self) -> int:
        return self.max_agents * self.agent_tokens_per_agent

    @property
    def total_tokens(self) -> int:
        return (
            self.map_tokens
            + self.agent_tokens
            + self.relation_tokens
            + self.event_tokens
            + self.scene_tokens
            + self.act_tokens
        )

    @property
    def coord_vocab_size(self) -> int:
        # 0..core-1, OUTSIDE, PAD.
        return self.core_map_size + 2

    @property
    def goal_delta_vocab_size(self) -> int:
        # [-clip, clip] plus PAD.
        return 2 * self.goal_delta_clip + 2

    @property
    def hops_vocab_size(self) -> int:
        # 0..max_hops, UNREACHABLE, CLIPPED/PAD.
        return self.max_hops + 3

    def validate(self) -> None:
        if self.local_map_size != self.core_map_size + 2:
            raise ValueError("local_map_size must equal core_map_size + 2 halo cells")
        if self.core_map_size % self.patch_size != 0:
            raise ValueError("core_map_size must be divisible by patch_size")
        if self.patches_per_side**2 != self.map_tokens:
            raise ValueError("map_tokens must equal (core_map_size / patch_size)^2")
        if self.history_tokens_per_agent != 2:
            raise ValueError("this implementation fixes two history summary tokens per agent")
        if self.event_tokens != self.history_steps:
            raise ValueError("event_tokens must equal history_steps")
        if self.num_actions != 5:
            raise ValueError("action order is fixed to WAIT, UP, DOWN, LEFT, RIGHT")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.total_tokens != 256:
            raise ValueError(
                "the architecture is defined as a 256-token policy context; "
                f"current configuration produces {self.total_tokens}"
            )


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    val_batch_size: int = 128
    gradient_accumulation_steps: int = 16
    epochs: int = 10
    max_steps: int | None = None
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 2000
    weight_decay: float = 1.0e-2
    beta1: float = 0.9
    beta2: float = 0.95
    use_scheduler: bool = True
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    amp: bool = True
    output_dir: str = "runs/base"

    ego_action_weight: float = 1.0
    neighbor_action_weight: float = 0.35
    act_request_weight: float = 0.5
    map_loss_weight: float = 0.02
    conflict_loss_weight: float = 0.10
    reason_loss_weight: float = 0.10
    scene_risk_loss_weight: float = 0.05


@dataclass(slots=True)
class DataConfig:
    kind: str = "synthetic"  # synthetic | npz_manifest | packed_policy
    train_samples: int = 4096
    val_samples: int = 512
    train_manifest: str | None = None
    val_manifest: str | None = None
    packed_train_manifest: str | None = None
    packed_val_manifest: str | None = None
    samples_per_epoch: int = 20000
    coordinate_order: str = "row_col"  # row_col | xy
    task_mode: str = "one_shot"  # one_shot | lifelong
    target_mode: str = "stay"  # stay | disappear
    goal_wait_keep_ratio: float = 0.2
    max_train_samples: int | None = None
    max_val_samples: int | None = 65536


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
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {"model", "training", "data"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown top-level config sections: {unknown}")
    return ProjectConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
    )
