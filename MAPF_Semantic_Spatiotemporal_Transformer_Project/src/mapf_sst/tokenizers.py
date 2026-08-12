from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .constants import Action, DeltaCTG
from .types import PolicyBatch


def signed_to_index(values: torch.Tensor, clip: int, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Map signed integers to [0,2*clip] plus one PAD index."""

    index = values.clamp(-clip, clip) + clip
    pad = 2 * clip + 1
    if valid is not None:
        index = torch.where(valid, index, torch.full_like(index, pad))
    return index.long()


def hops_to_index(values: torch.Tensor, max_hops: int, valid: torch.Tensor | None = None) -> torch.Tensor:
    """0..max_hops, UNREACHABLE=max+1, OVERFLOW/PAD=max+2."""

    index = values.clamp(0, max_hops)
    index = torch.where(values < 0, torch.full_like(index, max_hops + 1), index)
    index = torch.where(values > max_hops, torch.full_like(index, max_hops + 2), index)
    if valid is not None:
        index = torch.where(valid, index, torch.full_like(index, max_hops + 2))
    return index.long()


class PairSemanticEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        half = d_model // 2
        self.x_embedding = nn.Embedding(vocab_size, half)
        self.y_embedding = nn.Embedding(vocab_size, half)
        self.projection = nn.Sequential(
            nn.Linear(2 * half, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x_index: torch.Tensor, y_index: torch.Tensor) -> torch.Tensor:
        return self.projection(
            torch.cat((self.x_embedding(x_index), self.y_embedding(y_index)), dim=-1)
        )


class CurrentAgentTokenizer(nn.Module):
    """Current agent -> [P, G, R, WAIT, UP, DOWN, LEFT, RIGHT]."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.position_encoder = PairSemanticEncoder(config.current_coord_vocab, d, config.dropout)
        self.goal_encoder = PairSemanticEncoder(config.goal_delta_vocab, d, config.dropout)
        self.hops_embedding = nn.Embedding(config.hops_vocab, d)
        self.hops_norm = nn.LayerNorm(d)

        self.candidate_hops = nn.Embedding(config.hops_vocab, 64)
        self.candidate_delta = nn.Embedding(len(DeltaCTG), 32)
        self.greedy_embedding = nn.Embedding(2, 16)
        self.static_embedding = nn.Embedding(2, 16)
        self.in_core_embedding = nn.Embedding(2, 16)
        self.bottleneck_embedding = nn.Embedding(2, 16)
        self.candidate_projection = nn.Sequential(
            nn.Linear(160, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        self.role_embedding = nn.Embedding(2, d)  # ego, neighbor
        self.reset_embedding = nn.Embedding(2, d)
        self.clip_embedding = nn.Embedding(2, d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, batch: PolicyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        b, n = batch.current_valid.shape
        valid_xy = batch.current_valid[..., None].expand(-1, -1, 2)
        pos_clipped = batch.current_xy.abs() > cfg.current_coord_clip
        px = signed_to_index(batch.current_xy[..., 0], cfg.current_coord_clip, batch.current_valid)
        py = signed_to_index(batch.current_xy[..., 1], cfg.current_coord_clip, batch.current_valid)
        gx = signed_to_index(
            batch.current_goal_delta[..., 0], cfg.goal_delta_clip, batch.current_valid
        )
        gy = signed_to_index(
            batch.current_goal_delta[..., 1], cfg.goal_delta_clip, batch.current_valid
        )
        position = self.position_encoder(px, py)
        goal = self.goal_encoder(gx, gy)
        hops = self.hops_norm(
            self.hops_embedding(hops_to_index(batch.current_hops, cfg.max_hops, batch.current_valid))
        )

        candidate_hops = self.candidate_hops(
            hops_to_index(
                batch.candidate_one_hop_hops,
                cfg.max_hops,
                batch.current_valid[:, :, None].expand(-1, -1, cfg.num_actions),
            )
        )
        candidate = self.candidate_projection(
            torch.cat(
                (
                    candidate_hops,
                    self.candidate_delta(batch.candidate_delta_ctg.long()),
                    self.greedy_embedding(batch.candidate_greedy.long()),
                    self.static_embedding(batch.candidate_static_free.long()),
                    self.in_core_embedding(batch.candidate_in_core.long()),
                    self.bottleneck_embedding(batch.candidate_bottleneck.long()),
                ),
                dim=-1,
            )
        )

        role_ids = torch.ones(n, dtype=torch.long, device=batch.local_maps.device)
        role_ids[0] = 0
        role = self.role_embedding(role_ids)[None, :, :]
        reset = self.reset_embedding(batch.current_track_reset.long())
        clip_flag = self.clip_embedding(pos_clipped.any(dim=-1).long())
        shared = role + reset
        position = position + shared + clip_flag
        goal = goal + shared
        hops = hops + shared
        candidate = candidate + shared[:, :, None, :]

        tokens = torch.cat(
            (position[:, :, None, :], goal[:, :, None, :], hops[:, :, None, :], candidate),
            dim=2,
        )
        tokens = self.output_norm(tokens)
        tokens = tokens.masked_fill((~batch.current_valid)[:, :, None, None], 0.0)
        return tokens, candidate


class HistoryTokenizer(nn.Module):
    """Factual history -> [P, G, R, Action-Outcome] per track and lag."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.position_encoder = PairSemanticEncoder(config.history_coord_vocab, d, config.dropout)
        self.goal_encoder = PairSemanticEncoder(config.goal_delta_vocab, d, config.dropout)
        self.hops_embedding = nn.Embedding(config.hops_vocab, d)
        self.hops_norm = nn.LayerNorm(d)

        half = d // 2
        self.selected_embedding = nn.Embedding(len(Action), half)
        self.observed_embedding = nn.Embedding(len(Action), half)
        self.action_outcome_projection = nn.Sequential(
            nn.Linear(2 * half, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.lag_embedding = nn.Embedding(config.history_steps + 1, d)
        self.role_embedding = nn.Embedding(2, d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, batch: PolicyBatch) -> torch.Tensor:
        cfg = self.config
        b, h, t = batch.history_valid.shape
        valid_xy = batch.history_valid[..., None].expand(-1, -1, -1, 2)
        px = signed_to_index(batch.history_xy[..., 0], cfg.history_coord_clip, batch.history_valid)
        py = signed_to_index(batch.history_xy[..., 1], cfg.history_coord_clip, batch.history_valid)
        gx = signed_to_index(
            batch.history_goal_delta[..., 0], cfg.goal_delta_clip, batch.history_valid
        )
        gy = signed_to_index(
            batch.history_goal_delta[..., 1], cfg.goal_delta_clip, batch.history_valid
        )
        position = self.position_encoder(px, py)
        goal = self.goal_encoder(gx, gy)
        hops = self.hops_norm(
            self.hops_embedding(
                hops_to_index(batch.history_hops, cfg.max_hops, batch.history_valid)
            )
        )
        action_outcome = self.action_outcome_projection(
            torch.cat(
                (
                    self.selected_embedding(batch.history_selected_action.long()),
                    self.observed_embedding(batch.history_observed_move.long()),
                ),
                dim=-1,
            )
        )

        # Input lag axis is ordered [t-1, ..., t-history_steps].
        lag_ids = torch.arange(1, t + 1, device=batch.local_maps.device)
        lag = self.lag_embedding(lag_ids)[None, None, :, :]
        role_ids = torch.ones(h, dtype=torch.long, device=batch.local_maps.device)
        role_ids[0] = 0
        role = self.role_embedding(role_ids)[None, :, None, :]
        shared = lag + role

        tokens = torch.stack((position, goal, hops, action_outcome), dim=3)
        tokens = self.output_norm(tokens + shared[:, :, :, None, :])
        return tokens.masked_fill((~batch.history_valid)[:, :, :, None, None], 0.0)
