from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ProjectConfig


CHECKPOINT_FORMAT = 3


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ProjectConfig,
    epoch: int,
    step: int,
    metrics: dict[str, float] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT,
        "architecture": config.model.architecture,
        "model": model.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(
            "checkpoint is not format v3. The revised semantic-token layout is "
            "not shape/meaning compatible with the earlier Entity/Relation/ACT model."
        )
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
