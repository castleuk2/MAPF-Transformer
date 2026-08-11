from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .types import PolicyBatch


class AgentMicroEncoder(nn.Module):
    """Build one entity, two history, and five action-candidate tokens per agent."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        f = max(8, d // 8)

        self.coord_embedding = nn.Embedding(config.coord_vocab_size, f)
        self.goal_delta_embedding = nn.Embedding(config.goal_delta_vocab_size, f)
        self.hops_embedding = nn.Embedding(config.hops_vocab_size, f)
        self.bool_embedding = nn.Embedding(2, f)
        self.entity_projection = nn.Sequential(
            nn.Linear(8 * f, d), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(d, d), nn.LayerNorm(d),
        )

        self.slot_embedding = nn.Embedding(config.max_agents, d)
        self.role_embedding = nn.Embedding(2, d)
        self.valid_embedding = nn.Embedding(2, d)
        self.reset_embedding = nn.Embedding(2, d)
        self.agent_token_kind = nn.Embedding(config.agent_tokens_per_agent, d)

        self.action_embedding = nn.Embedding(config.num_actions + 1, f)
        self.outcome_embedding = nn.Embedding(config.outcome_states, f)
        self.history_delta_embedding = nn.Embedding(config.delta_ctg_states, f)
        self.history_step_projection = nn.Sequential(
            nn.Linear(4 * f, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.history_lag_embedding = nn.Embedding(config.history_steps, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_heads, dim_feedforward=d * config.mlp_ratio,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(
            layer, num_layers=config.history_layers, enable_nested_tensor=False
        )
        self.history_queries = nn.Parameter(torch.empty(2, d))
        nn.init.normal_(self.history_queries, std=0.02)
        self.history_pool = nn.MultiheadAttention(
            d, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.history_output_norm = nn.LayerNorm(d)

        self.candidate_action_embedding = nn.Embedding(config.num_actions, f)
        self.candidate_delta_embedding = nn.Embedding(config.delta_ctg_states, f)
        self.contender_embedding = nn.Embedding(config.contender_buckets, f)
        self.congestion_projection = nn.Sequential(nn.Linear(1, f), nn.Tanh())
        self.candidate_projection = nn.Sequential(
            nn.Linear(d + 12 * f, d * 2), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(d * 2, d), nn.LayerNorm(d),
        )

    def _slot_metadata(self, batch: PolicyBatch) -> torch.Tensor:
        b, n = batch.agent_valid.shape
        slot_ids = torch.arange(n, device=batch.agent_valid.device)
        role = torch.ones(b, n, dtype=torch.long, device=batch.agent_valid.device)
        role[:, 0] = 0
        return (
            self.slot_embedding(slot_ids)[None]
            + self.role_embedding(role)
            + self.valid_embedding(batch.agent_valid.long())
            + self.reset_embedding(batch.track_reset.long())
        )

    def _encode_entity(self, batch: PolicyBatch, metadata: torch.Tensor) -> torch.Tensor:
        xy = batch.agent_xy.clamp(0, self.config.coord_vocab_size - 1)
        goal = batch.goal_delta.clamp(-self.config.goal_delta_clip, self.config.goal_delta_clip)
        goal_index = (goal + self.config.goal_delta_clip).clamp(0, self.config.goal_delta_vocab_size - 2)
        hops = batch.remaining_hops.clamp(0, self.config.hops_vocab_size - 1)
        unreachable = (batch.remaining_hops == self.config.max_hops + 1).long()
        raw = torch.cat(
            (
                self.coord_embedding(xy[..., 0]),
                self.coord_embedding(xy[..., 1]),
                self.goal_delta_embedding(goal_index[..., 0]),
                self.goal_delta_embedding(goal_index[..., 1]),
                self.hops_embedding(hops),
                self.bool_embedding(batch.on_goal.long()),
                self.bool_embedding(batch.goal_outside.long()),
                self.bool_embedding(unreachable),
            ), dim=-1,
        )
        entity = self.entity_projection(raw) + metadata + self.agent_token_kind.weight[0]
        return entity.masked_fill((~batch.agent_valid)[..., None], 0.0)

    def _encode_history(self, batch: PolicyBatch, metadata: torch.Tensor) -> torch.Tensor:
        b, n, h = batch.history_selected.shape
        raw = torch.cat(
            (
                self.action_embedding(batch.history_selected.clamp(0, self.config.num_actions)),
                self.action_embedding(batch.history_executed.clamp(0, self.config.num_actions)),
                self.outcome_embedding(batch.history_outcome.clamp(0, self.config.outcome_states - 1)),
                self.history_delta_embedding(batch.history_delta_ctg.clamp(0, self.config.delta_ctg_states - 1)),
            ), dim=-1,
        )
        step = self.history_step_projection(raw)
        step = step + self.history_lag_embedding(torch.arange(h, device=step.device))[None, None]
        flat = step.reshape(b * n, h, self.config.d_model)
        valid = batch.history_valid.reshape(b * n, h) & batch.agent_valid.reshape(b * n, 1)
        padding = ~valid
        all_invalid = padding.all(dim=1)
        safe_padding = padding.clone()
        safe_padding[all_invalid, 0] = False
        flat = self.history_transformer(flat, src_key_padding_mask=safe_padding)
        queries = self.history_queries[None].expand(b * n, -1, -1)
        pooled, _ = self.history_pool(
            queries, flat, flat, key_padding_mask=safe_padding, need_weights=False
        )
        pooled[all_invalid] = 0.0
        pooled = pooled.reshape(b, n, 2, self.config.d_model)
        pooled = pooled + metadata[:, :, None, :]
        pooled[:, :, 0] = pooled[:, :, 0] + self.agent_token_kind.weight[1]
        pooled[:, :, 1] = pooled[:, :, 1] + self.agent_token_kind.weight[2]
        pooled = self.history_output_norm(pooled)
        return pooled.masked_fill((~batch.agent_valid)[:, :, None, None], 0.0)

    def _encode_candidates(
        self, batch: PolicyBatch, entity: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        b, n, a = batch.candidate_delta_ctg.shape
        action_ids = torch.arange(a, device=entity.device).view(1, 1, a).expand(b, n, -1)
        target = batch.candidate_target_xy.clamp(0, self.config.coord_vocab_size - 1)
        contender = batch.candidate_contenders.clamp(0, self.config.contender_buckets - 1)
        features = torch.cat(
            (
                self.candidate_action_embedding(action_ids),
                self.candidate_delta_embedding(batch.candidate_delta_ctg.clamp(0, self.config.delta_ctg_states - 1)),
                self.coord_embedding(target[..., 0]),
                self.coord_embedding(target[..., 1]),
                self.bool_embedding(batch.candidate_in_view.long()),
                self.bool_embedding(batch.candidate_greedy.long()),
                self.bool_embedding(batch.candidate_static_free.long()),
                self.bool_embedding(batch.candidate_target_occupied.long()),
                self.bool_embedding(batch.candidate_edge_swap.long()),
                self.bool_embedding(batch.candidate_bottleneck.long()),
                self.contender_embedding(contender),
                self.congestion_projection(batch.candidate_congestion.float().unsqueeze(-1)),
            ), dim=-1,
        )
        entity_expanded = entity[:, :, None, :].expand(-1, -1, a, -1)
        tokens = self.candidate_projection(torch.cat((entity_expanded, features), dim=-1))
        tokens = tokens + metadata[:, :, None, :] + self.agent_token_kind.weight[3 : 3 + a][None, None]
        return tokens.masked_fill((~batch.agent_valid)[:, :, None, None], 0.0)

    def forward(self, batch: PolicyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        metadata = self._slot_metadata(batch)
        entity = self._encode_entity(batch, metadata)
        history = self._encode_history(batch, metadata)
        candidates = self._encode_candidates(batch, entity, metadata)
        tokens = torch.cat((entity[:, :, None, :], history, candidates), dim=2)
        return tokens, entity, history, candidates
