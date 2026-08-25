from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, ProjectConfig
from .model import PreferenceCoordinationTransformer


def save_checkpoint(
    path: str | Path,
    model: PreferenceCoordinationTransformer,
    *,
    project_config: ProjectConfig,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": project_config.to_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PreferenceCoordinationTransformer, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = payload.get("config", {}).get("model", {})
    config = ModelConfig(**raw_model)
    model = PreferenceCoordinationTransformer(config)
    model.load_state_dict(payload["model"])
    return model, payload
