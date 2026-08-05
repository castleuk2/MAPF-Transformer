from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import LossConfig, ModelConfig
from .geometry import extract_core
from .model import MapEncoderOutput


@dataclass(slots=True)
class ReconstructionLossOutput:
    total: torch.Tensor
    cell_ce: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    port: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "cell_ce": float(self.cell_ce.detach().cpu()),
            "bce": float(self.bce.detach().cpu()),
            "dice": float(self.dice.detach().cpu()),
            "port": float(self.port.detach().cpu()),
        }


def occupancy_boundary_mask(target: torch.Tensor) -> torch.Tensor:
    """Return a 1-pixel morphological boundary mask for binary occupancy."""
    if target.ndim == 3:
        target_4d = target.unsqueeze(1)
    elif target.ndim == 4 and target.shape[1] == 1:
        target_4d = target
    else:
        raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(target.shape)}")
    target_4d = target_4d.to(torch.float32)
    dilation = F.max_pool2d(target_4d, kernel_size=3, stride=1, padding=1)
    erosion = -F.max_pool2d(-target_4d, kernel_size=3, stride=1, padding=1)
    boundary = (dilation - erosion).clamp(0.0, 1.0)
    return boundary.squeeze(1)


class MapReconstructionLoss(nn.Module):
    """Weighted occupancy reconstruction loss for evaluating the 25-token bottleneck."""

    def __init__(self, loss_config: LossConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.config = loss_config
        self.model_config = model_config

    def forward(self, output: MapEncoderOutput, halo_maps: torch.Tensor) -> ReconstructionLossOutput:
        if halo_maps.ndim == 2:
            halo_maps = halo_maps.unsqueeze(0)
        target_states = extract_core(halo_maps).to(device=output.reconstruction_logits.device)
        if self.config.mode == "cell_ce":
            logits = output.reconstruction_logits
            if logits.ndim != 4 or logits.shape[-1] != self.model_config.num_cell_states:
                raise ValueError("cell_ce expects reconstruction logits [B,H,W,num_cell_states].")
            cell_ce = F.cross_entropy(
                logits.reshape(-1, self.model_config.num_cell_states),
                target_states.long().reshape(-1),
            )
            zero = cell_ce.new_zeros(())
            return ReconstructionLossOutput(total=cell_ce, cell_ce=cell_ce, bce=zero, dice=zero, port=zero)
        target = target_states.eq(self.model_config.occupied_state).to(output.reconstruction_logits.dtype)
        logits = output.reconstruction_logits

        pos_weight = torch.as_tensor(
            self.config.occupied_pos_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        bce_map = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
        if self.config.boundary_pixel_weight > 1.0:
            boundary = occupancy_boundary_mask(target).to(dtype=logits.dtype)
            pixel_weight = 1.0 + (self.config.boundary_pixel_weight - 1.0) * boundary
            bce_map = bce_map * pixel_weight
        bce = bce_map.mean()

        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * target).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
        dice_score = (2.0 * intersection + self.config.eps) / (denominator + self.config.eps)
        dice = 1.0 - dice_score.mean()

        port = logits.new_zeros(())
        if self.config.port_weight > 0.0:
            if output.port_logits is None:
                raise ValueError("port_weight > 0 requires model.reconstruct_ports=true.")
            port_target = output.patch_geometry.port_open.to(
                device=output.port_logits.device,
                dtype=output.port_logits.dtype,
            ).reshape_as(output.port_logits)
            port_known = output.patch_geometry.port_known.to(
                device=output.port_logits.device,
                dtype=output.port_logits.dtype,
            ).reshape_as(output.port_logits)
            port_map = F.binary_cross_entropy_with_logits(output.port_logits, port_target, reduction="none")
            port = (port_map * port_known).sum() / port_known.sum().clamp_min(1.0)

        total = (
            self.config.bce_weight * bce
            + self.config.dice_weight * dice
            + self.config.port_weight * port
        )
        return ReconstructionLossOutput(total=total, cell_ce=total.new_zeros(()), bce=bce, dice=dice, port=port)
