from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig, ModelConfig, experiment_config_from_dict
from .model import MAPFTransformer


def save_checkpoint(
    path: str | Path,
    model: MAPFTransformer,
    experiment_config: ExperimentConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "experiment_config": experiment_config.to_dict(),
        "model_config": asdict(model.config),
        "step": int(step),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "training_state": training_state or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Not a MAPF Transformer checkpoint: {path}")
    return payload


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[MAPFTransformer, dict[str, Any]]:
    payload = load_checkpoint_payload(path, map_location=device)
    if "experiment_config" in payload:
        config = experiment_config_from_dict(payload["experiment_config"]).model
    else:
        config = ModelConfig(**payload["model_config"])
        config.validate()
    model = MAPFTransformer(config)
    model.load_state_dict(payload["model_state"], strict=strict)
    model.to(device)
    model.eval()
    return model, payload
