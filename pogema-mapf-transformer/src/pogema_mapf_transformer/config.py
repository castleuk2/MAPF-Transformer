from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class DatasetGenerationConfig:
    output_dir: str = "data/pogema_train"
    episodes: int = 100
    seed: int = 0
    num_agents: int = 16
    map_size: int = 21
    density: float = 0.25
    max_episode_steps: int = 128
    observation_type: str = "MAPF"
    obs_radius: int = 7
    on_target: str = "nothing"
    collision_system: str = "soft"
    expert: str = "mapf_lns2"
    planner_horizon: int = 192
    planner_retries: int = 16
    external_planner_command: list[str] | None = None
    mapf_lns2_binary: str = "external/MAPF-LNS2/lns"
    mapf_lns2_cutoff_time: float = 10.0
    mapf_lns2_init_algo: str = "PP"
    mapf_lns2_replan_algo: str = "PP"
    mapf_lns2_destroy_strategy: str = "Adaptive"
    mapf_lns2_neighbor_size: int = 8
    mapf_lns2_max_iterations: int = 0
    mapf_lns2_screen: int = 0
    precompute_tracking: bool = True
    tracking_grace_steps: int = 1
    compress: bool = True
    max_scenario_attempts: int = 20

    def validate(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.map_size <= 2:
            raise ValueError("map_size must be greater than two")
        if not 0 <= self.density < 1:
            raise ValueError("density must be in [0,1)")
        if self.observation_type != "MAPF":
            raise ValueError("The adapter requires observation_type='MAPF'")
        if self.obs_radius != 7:
            raise ValueError("The proposed 15x15 local map requires obs_radius=7")
        if self.collision_system not in {"priority", "block_both", "soft"}:
            raise ValueError("Unsupported POGEMA collision_system")
        if self.expert not in {"mapf_lns2", "prioritized", "external"}:
            raise ValueError("expert must be 'mapf_lns2', 'prioritized' or 'external'")
        if self.expert == "external" and not self.external_planner_command:
            raise ValueError("external expert requires external_planner_command")


@dataclass(slots=True)
class EvaluationConfig:
    checkpoint: str = "runs/mapf_transformer_base/best.pt"
    episodes: int = 10
    seed: int = 1000
    num_agents: int = 16
    map_size: int = 21
    density: float = 0.25
    max_episode_steps: int = 128
    collision_system: str = "soft"
    device: str = "auto"
    sample_actions: bool = False
    temperature: float = 1.0
    render_mode: str | None = None
    save_svg_dir: str | None = None

    def validate(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.collision_system not in {"priority", "block_both", "soft"}:
            raise ValueError("Unsupported POGEMA collision_system")
        if self.render_mode not in {None, "ansi"}:
            raise ValueError("render_mode must be null or 'ansi'")


def _load_dataclass(cls: type, values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    result = cls(**dict(values))
    result.validate()
    return result


def load_dataset_config(path: str | Path) -> DatasetGenerationConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    return _load_dataclass(DatasetGenerationConfig, values)


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    return _load_dataclass(EvaluationConfig, values)


def save_config(config: DatasetGenerationConfig | EvaluationConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(config), stream, sort_keys=False)
