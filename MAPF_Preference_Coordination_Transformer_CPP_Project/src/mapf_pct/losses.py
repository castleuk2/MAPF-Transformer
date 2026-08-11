from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainingConfig
from .model import PolicyOutput
from .types import PolicyBatch


@dataclass(slots=True)
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def compute_loss(output: PolicyOutput, batch: PolicyBatch, model_config: ModelConfig, training_config: TrainingConfig) -> LossOutput:
    if batch.all_agent_actions is None:
        raise ValueError("all_agent_actions is required for supervised training")
    labels = batch.all_agent_actions.long().clamp(0, model_config.num_actions - 1)
    per_agent_ce = F.cross_entropy(
        output.all_agent_logits.reshape(-1, model_config.num_actions),
        labels.reshape(-1), reduction="none"
    ).view_as(labels)
    ego_loss = per_agent_ce[:, 0].mean()
    neighbor_mask = batch.agent_valid.clone(); neighbor_mask[:, 0] = False
    neighbor_loss = _masked_mean(per_agent_ce, neighbor_mask)
    request_loss = F.cross_entropy(output.ego_logits, labels[:, 0])
    action_loss = (
        training_config.ego_action_weight * ego_loss
        + training_config.neighbor_action_weight * neighbor_loss
        + training_config.act_request_weight * request_loss
    )
    components: dict[str, torch.Tensor] = {
        "action": action_loss, "ego_action": ego_loss,
        "neighbor_action": neighbor_loss, "act_request": request_loss,
    }
    total = action_loss

    if output.map_reconstruction_logits is not None:
        target = batch.local_maps[:, 1:-1, 1:-1].long().clamp(0, 1)
        map_loss = F.cross_entropy(output.map_reconstruction_logits.reshape(-1, 2), target.reshape(-1))
        components["map"] = map_loss
        total = total + training_config.map_loss_weight * map_loss

    if output.conflict_logits is not None and output.relation_labels is not None and output.relation_valid.any():
        ce = F.cross_entropy(
            output.conflict_logits.reshape(-1, model_config.conflict_classes),
            output.relation_labels.reshape(-1), reduction="none"
        ).view_as(output.relation_labels)
        conflict_loss = _masked_mean(ce, output.relation_valid)
        components["conflict"] = conflict_loss
        total = total + training_config.conflict_loss_weight * conflict_loss

    if output.reason_logits is not None and batch.reason_labels is not None:
        mask = batch.agent_valid if batch.reason_valid is None else (batch.agent_valid & batch.reason_valid)
        bce = F.binary_cross_entropy_with_logits(output.reason_logits, batch.reason_labels.float(), reduction="none").mean(dim=-1)
        reason_loss = _masked_mean(bce, mask)
        components["reason"] = reason_loss
        total = total + training_config.reason_loss_weight * reason_loss

    if output.scene_risk_logits is not None and batch.scene_risk_labels is not None:
        scene_loss = F.cross_entropy(output.scene_risk_logits, batch.scene_risk_labels.long())
        components["scene_risk"] = scene_loss
        total = total + training_config.scene_risk_loss_weight * scene_loss

    components["total"] = total
    return LossOutput(total=total, components=components)
