from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig

try:
    from mapf_map_transformer.config import (
        ModelConfig as StructuredMapConfig,
        experiment_config_from_dict,
    )
    from mapf_map_transformer.model import StructuredMapTransformer
except ModuleNotFoundError as exc:  # pragma: no cover - installation error path
    raise ModuleNotFoundError(
        "The exact mapf-structured-map-transformer package is required. Install it "
        "before this project, e.g. `python -m pip install -e "
        "../../mapf-structured-map-transformer --no-deps`."
    ) from exc


class StructuredPatchMapEncoder(nn.Module):
    """Exact pretrained Structured-25 encoder with the SST tuple interface.

    This adapter intentionally does not reimplement the map network.  It owns the
    upstream :class:`mapf_map_transformer.model.StructuredMapTransformer`, loads
    its native checkpoint with ``strict=True``, and exposes only its 25 latent
    tokens plus the optional 15x15 reconstruction logits.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        checkpoint_path = self._resolve_checkpoint(config.map_checkpoint)
        payload: dict[str, Any] | None = None
        if checkpoint_path is not None:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "config" not in payload or "model_state" not in payload:
                raise ValueError(
                    f"{checkpoint_path} is not a native mapf-structured-map-transformer checkpoint"
                )
            structured_cfg = experiment_config_from_dict(payload["config"]).model
        else:
            structured_cfg = self._default_structured_config(config)

        self._validate_contract(config, structured_cfg)
        self.encoder = StructuredMapTransformer(structured_cfg)
        if payload is not None:
            self.encoder.load_state_dict(payload["model_state"], strict=True)

        if config.freeze_map_encoder:
            self.freeze()

    @staticmethod
    def _resolve_checkpoint(value: str | None) -> Path | None:
        if value is None or not value.strip():
            return None
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"structured map checkpoint not found: {path}")
        return path

    @staticmethod
    def _default_structured_config(config: ModelConfig) -> StructuredMapConfig:
        return StructuredMapConfig(
            halo_size=config.local_map_size,
            core_size=config.core_map_size,
            patch_size=config.patch_size,
            d_model=config.d_model,
            patch_hidden_dim=config.d_model,
            num_heads=config.n_heads,
            num_layers=config.map_layers,
            ffn_multiplier=config.mlp_ratio,
            dropout=config.dropout,
            reconstruction_classes=2,
        )

    @staticmethod
    def _validate_contract(config: ModelConfig, structured: StructuredMapConfig) -> None:
        expected = {
            "halo_size": config.local_map_size,
            "core_size": config.core_map_size,
            "patch_size": config.patch_size,
            "num_patch_tokens": config.map_tokens,
            "d_model": config.d_model,
            "num_heads": config.n_heads,
            "num_layers": config.map_layers,
            "reconstruction_classes": 2,
        }
        actual = {
            "halo_size": structured.halo_size,
            "core_size": structured.core_size,
            "patch_size": structured.patch_size,
            "num_patch_tokens": structured.num_patch_tokens,
            "d_model": structured.d_model,
            "num_heads": structured.num_heads,
            "num_layers": structured.num_layers,
            "reconstruction_classes": structured.reconstruction_classes,
        }
        mismatch = {key: (expected[key], actual[key]) for key in expected if expected[key] != actual[key]}
        if mismatch:
            raise ValueError(f"SST/map checkpoint contract mismatch: {mismatch}")

    def freeze(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @property
    def is_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.encoder.parameters())

    def train(self, mode: bool = True) -> "StructuredPatchMapEncoder":
        super().train(mode)
        # A frozen encoder must also keep Dropout disabled during policy training.
        if self.config.freeze_map_encoder:
            self.encoder.eval()
        return self

    def extract_raw_features(self, local_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.encoder._build_geometry(local_maps)
        p = self.encoder.config.patch_size
        pps = self.encoder.config.patch_grid_size
        openings = geometry.port_open.view(local_maps.shape[0], pps, pps, 4, p)
        return geometry.features, openings

    def forward(
        self,
        local_maps: torch.Tensor,
        *,
        return_reconstruction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = self.encoder(local_maps)
        reconstruction = output.reconstruction_logits if return_reconstruction else None
        return output.latent_tokens, reconstruction
