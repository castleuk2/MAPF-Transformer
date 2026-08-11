from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .constants import TokenField
from .types import PolicyBatch


class StructuredAttentionBias(nn.Module):
    """Build semantic, temporal, spatial and conflict priors for 256 tokens."""

    def __init__(self, config: ModelConfig, field_ids: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("field_ids", field_ids.long(), persistent=False)
        h = config.n_heads
        f = int(TokenField.COUNT)
        self.field_pair_bias = nn.Parameter(torch.zeros(h, f, f))
        self.relative_time_bias = nn.Parameter(
            torch.zeros(h, 2 * config.history_steps + 1)
        )
        self.same_current_agent_bias = nn.Parameter(torch.zeros(h))
        self.same_history_track_bias = nn.Parameter(torch.zeros(h))
        self.current_history_track_bias = nn.Parameter(torch.zeros(h))
        self.vertex_bias = nn.Parameter(torch.zeros(h))
        self.edge_swap_bias = nn.Parameter(torch.zeros(h))
        self.occupancy_bias = nn.Parameter(torch.zeros(h))
        self.corridor_bias = nn.Parameter(torch.zeros(h))
        self.candidate_source_map_bias = nn.Parameter(torch.zeros(h))
        self.candidate_target_map_bias = nn.Parameter(torch.zeros(h))
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.01)

        length = config.total_tokens
        current_agent = torch.full((length,), -1, dtype=torch.long)
        history_track = torch.full((length,), -1, dtype=torch.long)
        time_lag = torch.zeros(length, dtype=torch.long)

        for agent in range(config.max_current_agents):
            start = config.current_offset + agent * config.current_tokens_per_agent
            current_agent[start : start + config.current_tokens_per_agent] = agent

        for lag in range(config.history_steps):
            for track in range(config.history_tracks):
                start = (
                    config.history_offset
                    + lag * config.history_tracks * config.history_tokens_per_step
                    + track * config.history_tokens_per_step
                )
                history_track[start : start + config.history_tokens_per_step] = track
                time_lag[start : start + config.history_tokens_per_step] = lag + 1

        candidate_indices = []
        candidate_agent = []
        candidate_action = []
        for agent in range(config.max_current_agents):
            start = config.current_offset + agent * config.current_tokens_per_agent + 3
            for action in range(config.num_actions):
                candidate_indices.append(start + action)
                candidate_agent.append(agent)
                candidate_action.append(action)

        self.register_buffer("current_agent", current_agent, persistent=False)
        self.register_buffer("history_track", history_track, persistent=False)
        self.register_buffer("time_lag", time_lag, persistent=False)
        self.register_buffer(
            "candidate_indices", torch.tensor(candidate_indices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "candidate_agent", torch.tensor(candidate_agent, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "candidate_action", torch.tensor(candidate_action, dtype=torch.long), persistent=False
        )

    def _static_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        field = self.field_ids.to(device)
        bias = self.field_pair_bias[:, field[:, None], field[None, :]]
        dt = self.time_lag[:, None] - self.time_lag[None, :]
        dt_index = (dt + self.config.history_steps).clamp(
            0, 2 * self.config.history_steps
        )
        bias = bias + self.relative_time_bias[:, dt_index]

        same_current = (
            (self.current_agent[:, None] >= 0)
            & (self.current_agent[:, None] == self.current_agent[None, :])
        )
        same_history = (
            (self.history_track[:, None] >= 0)
            & (self.history_track[:, None] == self.history_track[None, :])
        )
        bias = bias + self.same_current_agent_bias[:, None, None] * same_current[None]
        bias = bias + self.same_history_track_bias[:, None, None] * same_history[None]
        return bias.to(device=device, dtype=dtype)

    def forward(self, batch: PolicyBatch, *, dtype: torch.dtype) -> torch.Tensor:
        cfg = self.config
        b = batch.batch_size
        device = batch.local_maps.device
        base = self._static_bias(device, dtype).unsqueeze(0).expand(b, -1, -1, -1).clone()

        current_agent = self.current_agent.to(device)
        history_track = self.history_track.to(device)
        safe_track = history_track.clamp_min(0)
        track_slot = batch.history_track_current_slot[:, safe_track]
        current_to_history = (
            (current_agent[None, :, None] >= 0)
            & (history_track[None, None, :] >= 0)
            & (current_agent[None, :, None] == track_slot[:, None, :])
        )
        current_history = current_to_history | current_to_history.transpose(-1, -2)
        base += self.current_history_track_bias[None, :, None, None] * current_history[:, None]

        # Candidate-pair relations: all pairs are represented as score bias, so
        # there is no Top-K relation-token truncation.
        q = cfg.max_current_agents * cfg.num_actions
        agent = self.candidate_agent.to(device)
        action = self.candidate_action.to(device)
        targets = batch.candidate_target_core_xy.reshape(b, q, 2)
        sources_agent = batch.current_xy + cfg.core_map_size // 2
        sources = sources_agent[:, agent]
        valid = (
            batch.current_valid[:, :, None]
            & batch.candidate_static_free
        ).reshape(b, q)
        different_agent = agent[:, None] != agent[None, :]
        pair_valid = valid[:, :, None] & valid[:, None, :] & different_agent[None]

        same_target = (targets[:, :, None, :] == targets[:, None, :, :]).all(dim=-1)
        edge_swap = (
            (targets[:, :, None, :] == sources[:, None, :, :]).all(dim=-1)
            & (sources[:, :, None, :] == targets[:, None, :, :]).all(dim=-1)
        )
        occupancy = (targets[:, :, None, :] == sources[:, None, :, :]).all(dim=-1)
        bottleneck = batch.candidate_bottleneck.reshape(b, q)
        target_distance = (targets[:, :, None, :] - targets[:, None, :, :]).abs().sum(dim=-1)
        corridor = (
            bottleneck[:, :, None]
            & bottleneck[:, None, :]
            & (target_distance <= 1)
        )

        pair_bias = torch.zeros(b, cfg.n_heads, q, q, device=device, dtype=dtype)
        pair_bias += self.vertex_bias[None, :, None, None] * (same_target & pair_valid)[:, None]
        pair_bias += self.edge_swap_bias[None, :, None, None] * (edge_swap & pair_valid)[:, None]
        pair_bias += self.occupancy_bias[None, :, None, None] * (occupancy & pair_valid)[:, None]
        pair_bias += self.corridor_bias[None, :, None, None] * (corridor & pair_valid)[:, None]
        ci = self.candidate_indices.to(device)
        base[:, :, ci[:, None], ci[None, :]] += pair_bias

        # Candidate-to-map source/target anchors remain visible in the global
        # Transformer even after the dedicated spatial-fusion stage.
        pps = cfg.patches_per_side
        source_patch = torch.div(
            sources.clamp(0, cfg.core_map_size - 1),
            cfg.patch_size,
            rounding_mode="floor",
        )
        target_patch = torch.div(
            targets.clamp(0, cfg.core_map_size - 1),
            cfg.patch_size,
            rounding_mode="floor",
        )
        source_patch_idx = source_patch[..., 0] * pps + source_patch[..., 1]
        target_patch_idx = target_patch[..., 0] * pps + target_patch[..., 1]
        target_ok = batch.candidate_in_core.reshape(b, q)
        batch_index = torch.arange(b, device=device)[:, None, None]
        head_index = torch.arange(cfg.n_heads, device=device)[None, :, None]
        token_index = ci[None, None, :]
        source_index = source_patch_idx[:, None, :]
        target_index = target_patch_idx[:, None, :]
        source_value = valid[:, None, :].to(dtype) * self.candidate_source_map_bias[None, :, None]
        target_value = (valid & target_ok)[:, None, :].to(dtype) * self.candidate_target_map_bias[None, :, None]
        base[batch_index, head_index, token_index, source_index] += source_value
        base[batch_index, head_index, source_index, token_index] += source_value
        base[batch_index, head_index, token_index, target_index] += target_value
        base[batch_index, head_index, target_index, token_index] += target_value
        return base
