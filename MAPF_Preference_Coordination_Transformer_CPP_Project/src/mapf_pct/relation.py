from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig
from .constants import ConflictType
from .types import PolicyBatch


@dataclass(slots=True)
class RelationOutput:
    tokens: torch.Tensor
    pair_index: torch.Tensor
    valid: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor | None


class TopKRelationBuilder(nn.Module):
    """Constructs explicit tokens for the 24 highest-risk agent pairs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        pair_i, pair_j = torch.triu_indices(config.max_agents, config.max_agents, offset=1)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)
        d = config.d_model
        self.feature_projection = nn.Sequential(
            nn.Linear(config.relation_numeric_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(3 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.rank_embedding = nn.Embedding(config.relation_tokens, d)
        self.relation_type = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.relation_type, std=0.02)

    @staticmethod
    def _gather_pair(x: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, pair_i], x[:, pair_j]

    def _all_pair_features(self, batch: PolicyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j = self.pair_i, self.pair_j
        valid_i, valid_j = self._gather_pair(batch.agent_valid, i, j)
        pair_valid = valid_i & valid_j

        pos_i, pos_j = self._gather_pair(batch.agent_xy.float(), i, j)
        delta = pos_j - pos_i
        distance = delta.abs().sum(dim=-1)
        hops_i, hops_j = self._gather_pair(batch.remaining_hops.float(), i, j)
        wait_fraction = (
            ((batch.history_selected == 0) & batch.history_valid).float().sum(dim=-1)
            / batch.history_valid.float().sum(dim=-1).clamp_min(1.0)
        )
        wait_i, wait_j = self._gather_pair(wait_fraction, i, j)

        targets_i, targets_j = self._gather_pair(batch.candidate_target_xy, i, j)
        feasible = batch.candidate_static_free & batch.candidate_in_view & batch.agent_valid[:, :, None]
        feasible_i, feasible_j = self._gather_pair(feasible, i, j)
        greedy_i, greedy_j = self._gather_pair(batch.candidate_greedy & feasible, i, j)
        match = (targets_i[:, :, :, None, :] == targets_j[:, :, None, :, :]).all(dim=-1)
        feasible_match = match & feasible_i[:, :, :, None] & feasible_j[:, :, None, :]
        overlap_count = feasible_match.float().sum(dim=(-1, -2))
        greedy_overlap = (
            match & greedy_i[:, :, :, None] & greedy_j[:, :, None, :]
        ).any(dim=(-1, -2))

        # Whether any feasible candidates form a two-agent edge swap.
        ti_to_j = (targets_i[:, :, :, None, :] == pos_j[:, :, None, None, :].long()).all(dim=-1)
        tj_to_i = (targets_j[:, :, None, :, :] == pos_i[:, :, None, None, :].long()).all(dim=-1)
        edge_swap = (
            ti_to_j
            & tj_to_i
            & feasible_i[:, :, :, None]
            & feasible_j[:, :, None, :]
        ).any(dim=(-1, -2))

        bottleneck_i, bottleneck_j = self._gather_pair(batch.candidate_bottleneck, i, j)
        bottleneck_overlap = (
            feasible_match
            & bottleneck_i[:, :, :, None]
            & bottleneck_j[:, :, None, :]
        ).any(dim=(-1, -2))
        congestion_i, congestion_j = self._gather_pair(batch.candidate_congestion.float(), i, j)
        contender_i, contender_j = self._gather_pair(batch.candidate_contenders.float(), i, j)
        mean_congestion = 0.5 * (congestion_i.mean(dim=-1) + congestion_j.mean(dim=-1))
        max_contenders = torch.maximum(contender_i.max(dim=-1).values, contender_j.max(dim=-1).values)

        risk = (
            4.0 * greedy_overlap.float()
            + 4.0 * edge_swap.float()
            + 2.0 / (1.0 + distance)
            + 0.2 * overlap_count
            + 1.0 * bottleneck_overlap.float()
            + 0.5 * mean_congestion
            + 0.1 * max_contenders
        )
        risk = risk.masked_fill(~pair_valid, -1.0e9)

        norm_hops = float(max(1, self.config.max_hops))
        norm_coord = float(max(1, self.config.core_map_size - 1))
        features = torch.stack(
            (
                delta[..., 0] / norm_coord,
                delta[..., 1] / norm_coord,
                distance / (2.0 * norm_coord),
                (hops_j - hops_i).clamp(-norm_hops, norm_hops) / norm_hops,
                wait_j - wait_i,
                overlap_count / float(self.config.num_actions),
                greedy_overlap.float(),
                edge_swap.float(),
                bottleneck_overlap.float(),
                mean_congestion,
                max_contenders / float(max(1, self.config.contender_buckets - 1)),
                risk.clamp_min(0.0) / 10.0,
            ),
            dim=-1,
        )
        if features.shape[-1] != self.config.relation_numeric_dim:
            raise RuntimeError("relation feature dimensionality mismatch")
        return features, risk, pair_valid

    def _all_pair_labels(self, batch: PolicyBatch) -> torch.Tensor | None:
        if batch.all_agent_actions is None:
            return None
        actions = batch.all_agent_actions.clamp(0, self.config.num_actions - 1)
        selected_target = torch.gather(
            batch.candidate_target_xy,
            2,
            actions[:, :, None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        selected_bottleneck = torch.gather(
            batch.candidate_bottleneck,
            2,
            actions[:, :, None],
        ).squeeze(2)
        i, j = self.pair_i, self.pair_j
        ti, tj = self._gather_pair(selected_target, i, j)
        pi, pj = self._gather_pair(batch.agent_xy, i, j)
        bi, bj = self._gather_pair(selected_bottleneck, i, j)
        labels = torch.full(ti.shape[:2], int(ConflictType.NONE), device=ti.device, dtype=torch.long)
        vertex = (ti == tj).all(dim=-1)
        edge = (ti == pj).all(dim=-1) & (tj == pi).all(dim=-1)
        corridor = (bi | bj) & ((pi - pj).abs().sum(dim=-1) <= 2)
        labels = torch.where(corridor, torch.full_like(labels, int(ConflictType.CORRIDOR)), labels)
        labels = torch.where(vertex, torch.full_like(labels, int(ConflictType.VERTEX)), labels)
        labels = torch.where(edge, torch.full_like(labels, int(ConflictType.EDGE_SWAP)), labels)
        return labels

    def forward(self, entity_tokens: torch.Tensor, batch: PolicyBatch) -> RelationOutput:
        features, risk, pair_valid = self._all_pair_features(batch)
        k = self.config.relation_tokens
        top_values, top_index = torch.topk(risk, k=k, dim=1, largest=True, sorted=True)
        selected_valid = torch.gather(pair_valid, 1, top_index) & (top_values > -1.0e8)
        selected_features = torch.gather(
            features,
            1,
            top_index[..., None].expand(-1, -1, features.shape[-1]),
        )
        selected_i = self.pair_i[top_index]
        selected_j = self.pair_j[top_index]
        pair_index = torch.stack((selected_i, selected_j), dim=-1)

        d = entity_tokens.shape[-1]
        entity_i = torch.gather(entity_tokens, 1, selected_i[..., None].expand(-1, -1, d))
        entity_j = torch.gather(entity_tokens, 1, selected_j[..., None].expand(-1, -1, d))
        feature_token = self.feature_projection(selected_features)
        token = self.token_projection(torch.cat((entity_i, entity_j, feature_token), dim=-1))
        rank = self.rank_embedding(torch.arange(k, device=token.device))
        token = token + rank[None] + self.relation_type
        token = token.masked_fill((~selected_valid)[..., None], 0.0)

        labels = self._all_pair_labels(batch)
        if labels is not None:
            labels = torch.gather(labels, 1, top_index)
        return RelationOutput(
            tokens=token,
            pair_index=pair_index,
            valid=selected_valid,
            features=selected_features,
            labels=labels,
        )
