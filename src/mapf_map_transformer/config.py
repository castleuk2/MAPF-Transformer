from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class ModelConfig:
    halo_size: int = 17
    core_size: int = 15
    patch_size: int = 3
    num_cell_states: int = 2
    free_state: int = 0
    occupied_state: int = 1
    unknown_state: int | None = None
    d_model: int = 256
    patch_hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 1
    ffn_multiplier: int = 4
    dropout: float = 0.1
    position_encoding: str = "learned"  # learned | sincos
    use_center_embedding: bool = True
    use_relative_position_bias: bool = True
    use_connectivity_bias: bool = True
    include_port_openings: bool = True
    include_outer_edge_mask: bool = True
    include_port_known_mask: bool = False
    final_norm: bool = True
    reconstruct_ports: bool = False

    def validate(self) -> None:
        if self.halo_size != self.core_size + 2:
            raise ValueError("halo_size must be core_size + 2 (one-cell halo on each side).")
        if self.core_size % self.patch_size != 0:
            raise ValueError("core_size must be divisible by patch_size.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.position_encoding not in {"learned", "sincos"}:
            raise ValueError("position_encoding must be 'learned' or 'sincos'.")
        if self.position_encoding == "sincos" and self.d_model % 4 != 0:
            raise ValueError("2D sine/cosine encoding requires d_model divisible by 4.")
        if self.num_cell_states < 2:
            raise ValueError("num_cell_states must be at least 2.")
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative.")

    @property
    def patch_grid_size(self) -> int:
        return self.core_size // self.patch_size

    @property
    def num_patch_tokens(self) -> int:
        return self.patch_grid_size**2

    @property
    def patch_feature_dim(self) -> int:
        dim = self.patch_size * self.patch_size * self.num_cell_states
        if self.include_port_openings:
            dim += 4 * self.patch_size
        if self.include_outer_edge_mask:
            dim += 4
        if self.include_port_known_mask:
            dim += 4 * self.patch_size
        return dim


@dataclass(slots=True)
class LossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 0.5
    port_weight: float = 0.0
    occupied_pos_weight: float = 2.0
    boundary_pixel_weight: float = 2.0
    eps: float = 1.0e-6


@dataclass(slots=True)
class DatasetConfig:
    kind: str = "synthetic"  # synthetic | npz
    train_path: str | None = None
    val_path: str | None = None
    train_samples: int = 20000
    val_samples: int = 2000
    seed: int = 7
    min_density: float = 0.08
    max_density: float = 0.38
    pattern_mix: tuple[str, ...] = ("bernoulli", "walls", "rooms", "corridors", "mixed")
    ensure_center_free: bool = True


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "runs/map_reconstruction"
    epochs: int = 30
    batch_size: int = 128
    val_batch_size: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 200
    max_steps: int | None = None
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    seed: int = 42
    log_interval: int = 20
    eval_interval_epochs: int = 1
    checkpoint_interval_epochs: int = 1
    save_visualizations: int = 8


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.model.validate()
        if self.training.batch_size <= 0 or self.training.val_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")
        if self.training.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.loss.occupied_pos_weight <= 0:
            raise ValueError("occupied_pos_weight must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_dataclass(cls: type[Any], values: Mapping[str, Any] | None) -> Any:
    return cls(**dict(values or {}))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = ExperimentConfig(
        model=_construct_dataclass(ModelConfig, raw.get("model")),
        loss=_construct_dataclass(LossConfig, raw.get("loss")),
        dataset=_construct_dataclass(DatasetConfig, raw.get("dataset")),
        training=_construct_dataclass(TrainingConfig, raw.get("training")),
    )
    # YAML lists are accepted for tuple-annotated fields.
    if isinstance(cfg.dataset.pattern_mix, list):
        cfg.dataset.pattern_mix = tuple(cfg.dataset.pattern_mix)
    cfg.validate()
    return cfg


def experiment_config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig(
        model=_construct_dataclass(ModelConfig, raw.get("model")),
        loss=_construct_dataclass(LossConfig, raw.get("loss")),
        dataset=_construct_dataclass(DatasetConfig, raw.get("dataset")),
        training=_construct_dataclass(TrainingConfig, raw.get("training")),
    )
    if isinstance(cfg.dataset.pattern_mix, list):
        cfg.dataset.pattern_mix = tuple(cfg.dataset.pattern_mix)
    cfg.validate()
    return cfg


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
