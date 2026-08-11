from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainingConfig
from .tokenizers import hops_to_index, signed_to_index
from .types import PolicyBatch, PolicyOutput


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    ego_action: torch.Tensor
    ranking: torch.Tensor
    map_reconstruction: torch.Tensor
    semantic_reconstruction: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "ego_action": float(self.ego_action.detach()),
            "ranking": float(self.ranking.detach()),
            "map_reconstruction": float(self.map_reconstruction.detach()),
            "semantic_reconstruction": float(self.semantic_reconstruction.detach()),
        }


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def compute_loss(
    output: PolicyOutput,
    batch: PolicyBatch,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
) -> LossBreakdown:
    ego = F.cross_entropy(output.ego_logits, batch.ego_action.long())

    ranking = output.ego_logits.sum() * 0.0
    if batch.action_soft_targets is not None and train_cfg.ranking_loss_weight > 0:
        target = batch.action_soft_targets.float()
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        ranking = F.kl_div(
            F.log_softmax(output.ego_logits, dim=-1), target, reduction="batchmean"
        )

    map_loss = output.ego_logits.sum() * 0.0
    if output.map_reconstruction_logits is not None and train_cfg.map_loss_weight > 0:
        target_map = batch.local_maps[:, 1:-1, 1:-1].long().clamp(0, 1)
        map_loss = F.cross_entropy(
            output.map_reconstruction_logits.reshape(-1, 2), target_map.reshape(-1)
        )

    semantic_loss = output.ego_logits.sum() * 0.0
    recon = output.semantic_reconstruction
    if recon is not None and train_cfg.semantic_reconstruction_weight > 0:
        current_valid = batch.current_valid.bool()
        current_x = signed_to_index(
            batch.current_xy[..., 0], model_cfg.current_coord_clip, current_valid
        )
        current_y = signed_to_index(
            batch.current_xy[..., 1], model_cfg.current_coord_clip, current_valid
        )
        goal_x = signed_to_index(
            batch.current_goal_delta[..., 0], model_cfg.goal_delta_clip, current_valid
        )
        goal_y = signed_to_index(
            batch.current_goal_delta[..., 1], model_cfg.goal_delta_clip, current_valid
        )
        current_hops = hops_to_index(
            batch.current_hops, model_cfg.max_hops, current_valid
        )
        components = [
            _masked_ce(recon.current_position_x, current_x, current_valid),
            _masked_ce(recon.current_position_y, current_y, current_valid),
            _masked_ce(recon.current_goal_x, goal_x, current_valid),
            _masked_ce(recon.current_goal_y, goal_y, current_valid),
            _masked_ce(recon.current_hops, current_hops, current_valid),
        ]

        # Model history outputs are [B,T,H,*]; batch storage is [B,H,T].
        history_valid = batch.history_valid.permute(0, 2, 1).bool()
        history_x = signed_to_index(
            batch.history_xy[..., 0].permute(0, 2, 1),
            model_cfg.history_coord_clip,
            history_valid,
        )
        history_y = signed_to_index(
            batch.history_xy[..., 1].permute(0, 2, 1),
            model_cfg.history_coord_clip,
            history_valid,
        )
        history_goal_x = signed_to_index(
            batch.history_goal_delta[..., 0].permute(0, 2, 1),
            model_cfg.goal_delta_clip,
            history_valid,
        )
        history_goal_y = signed_to_index(
            batch.history_goal_delta[..., 1].permute(0, 2, 1),
            model_cfg.goal_delta_clip,
            history_valid,
        )
        history_hops = hops_to_index(
            batch.history_hops.permute(0, 2, 1),
            model_cfg.max_hops,
            history_valid,
        )
        selected = batch.history_selected_action.permute(0, 2, 1).long()
        observed = batch.history_observed_move.permute(0, 2, 1).long()
        components.extend(
            [
                _masked_ce(recon.history_position_x, history_x, history_valid),
                _masked_ce(recon.history_position_y, history_y, history_valid),
                _masked_ce(recon.history_goal_x, history_goal_x, history_valid),
                _masked_ce(recon.history_goal_y, history_goal_y, history_valid),
                _masked_ce(recon.history_hops, history_hops, history_valid),
                _masked_ce(recon.history_selected_action, selected, history_valid),
                _masked_ce(recon.history_observed_move, observed, history_valid),
            ]
        )
        semantic_loss = torch.stack(components).mean()

    total = (
        train_cfg.ego_action_weight * ego
        + train_cfg.ranking_loss_weight * ranking
        + train_cfg.map_loss_weight * map_loss
        + train_cfg.semantic_reconstruction_weight * semantic_loss
    )
    return LossBreakdown(
        total=total,
        ego_action=ego,
        ranking=ranking,
        map_reconstruction=map_loss,
        semantic_reconstruction=semantic_loss,
    )
