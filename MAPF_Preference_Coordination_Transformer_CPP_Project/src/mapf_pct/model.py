from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .agent_encoder import AgentMicroEncoder
from .attention import TypeBiasedTransformerBlock
from .config import ModelConfig
from .context_encoder import EventEncoder, SceneTokenEncoder
from .map_encoder import CandidateToMapCrossAttention, StructuredPatchMapEncoder
from .relation import RelationOutput, TopKRelationBuilder
from .types import PolicyBatch


class TokenType:
    MAP = 0
    ENTITY = 1
    HISTORY = 2
    CANDIDATE = 3
    RELATION = 4
    EVENT = 5
    SCENE = 6
    ACT = 7
    COUNT = 8


@dataclass(slots=True)
class PolicyOutput:
    all_agent_logits: torch.Tensor
    ego_logits: torch.Tensor
    ego_request_delta: torch.Tensor
    reason_logits: torch.Tensor | None
    conflict_logits: torch.Tensor | None
    scene_risk_logits: torch.Tensor | None
    map_reconstruction_logits: torch.Tensor | None
    relation_index: torch.Tensor
    relation_valid: torch.Tensor
    relation_labels: torch.Tensor | None
    scene_embedding: torch.Tensor
    token_padding_mask: torch.Tensor
    final_tokens: torch.Tensor | None = None


class PreferenceCoordinationTransformer(nn.Module):
    """Fixed 256-token, bidirectional Preference-Coordination policy.

    Token budget:
      25 map + 25*(1 entity + 2 history + 5 candidates)
      + 24 relations + 5 events + 1 scene + 1 ACT = 256.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        config.validate()
        self.map_encoder = StructuredPatchMapEncoder(config)
        self._map_encoder_frozen = bool(config.freeze_map_encoder)
        if config.map_checkpoint is not None:
            checkpoint_path = Path(config.map_checkpoint)
            if not checkpoint_path.exists() and not checkpoint_path.is_absolute():
                checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint.get("state_dict")))
            if state is None:
                raise KeyError("map checkpoint must contain model_state, model_state_dict, or state_dict")
            self.map_encoder.load_state_dict(state, strict=True)
        if self._map_encoder_frozen:
            if config.map_checkpoint is None:
                raise ValueError("freeze_map_encoder=True requires map_checkpoint")
            self.map_encoder.requires_grad_(False)
            self.map_encoder.eval()
        self.agent_encoder = AgentMicroEncoder(config)
        self.candidate_map_attention = CandidateToMapCrossAttention(config)
        self.relation_builder = TopKRelationBuilder(config)
        self.event_encoder = EventEncoder(config)
        self.scene_encoder = SceneTokenEncoder(config)

        self.act_query = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.act_query, std=0.02)
        self.token_type_embedding = nn.Embedding(TokenType.COUNT, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.coordination_blocks = nn.ModuleList(
            [
                TypeBiasedTransformerBlock(
                    config.d_model,
                    config.n_heads,
                    TokenType.COUNT,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.coordination_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)

        self.candidate_score_head = nn.Linear(config.d_model, 1)
        self.act_head = nn.Linear(config.d_model, config.num_actions)
        self.reason_head = nn.Linear(config.d_model, config.reason_classes) if config.enable_reason_head else None
        self.conflict_head = nn.Linear(config.d_model, config.conflict_classes) if config.enable_conflict_head else None
        self.scene_risk_head = nn.Linear(config.d_model, config.scene_risk_classes) if config.enable_scene_risk_head else None

        token_type_ids = self._build_token_type_ids()
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)
        if token_type_ids.numel() != config.total_tokens:
            raise RuntimeError("token type layout does not match configured token budget")

    def train(self, mode: bool = True):
        super().train(mode)
        if self._map_encoder_frozen:
            # Frozen dropout must stay disabled so identical maps produce stable tokens.
            self.map_encoder.eval()
        return self

    def _build_token_type_ids(self) -> torch.Tensor:
        cfg = self.config
        types: list[int] = [TokenType.MAP] * cfg.map_tokens
        per_agent = [TokenType.ENTITY, TokenType.HISTORY, TokenType.HISTORY] + [TokenType.CANDIDATE] * cfg.num_actions
        for _ in range(cfg.max_agents):
            types.extend(per_agent)
        types.extend([TokenType.RELATION] * cfg.relation_tokens)
        types.extend([TokenType.EVENT] * cfg.event_tokens)
        types.extend([TokenType.SCENE, TokenType.ACT])
        return torch.tensor(types, dtype=torch.long)

    def _assemble(
        self,
        map_tokens: torch.Tensor,
        agent_tokens: torch.Tensor,
        relation: RelationOutput,
        event_tokens: torch.Tensor,
        scene_token: torch.Tensor,
        batch: PolicyBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = map_tokens.shape[0]
        agent_flat = agent_tokens.reshape(b, self.config.agent_tokens, self.config.d_model)
        act = self.act_query.view(1, 1, -1).expand(b, -1, -1) + agent_tokens[:, 0, 0:1]
        tokens = torch.cat((map_tokens, agent_flat, relation.tokens, event_tokens, scene_token, act), dim=1)
        if tokens.shape[1] != self.config.total_tokens:
            raise RuntimeError(f"expected 256 tokens, received {tokens.shape[1]}")

        map_padding = torch.zeros(b, self.config.map_tokens, dtype=torch.bool, device=tokens.device)
        agent_padding = (~batch.agent_valid)[:, :, None].expand(-1, -1, self.config.agent_tokens_per_agent).reshape(b, self.config.agent_tokens)
        special_padding = torch.zeros(b, 2, dtype=torch.bool, device=tokens.device)
        padding = torch.cat((map_padding, agent_padding, ~relation.valid, ~batch.event_valid, special_padding), dim=1)
        tokens = self.input_norm(tokens + self.token_type_embedding(self.token_type_ids.to(tokens.device))[None])
        return tokens.masked_fill(padding[..., None], 0.0), padding

    def forward(
        self,
        batch: PolicyBatch,
        *,
        return_tokens: bool = False,
        return_map_reconstruction: bool | None = None,
    ) -> PolicyOutput:
        if return_map_reconstruction is None:
            return_map_reconstruction = self.config.enable_map_reconstruction and self.training
        map_tokens, reconstruction = self.map_encoder(batch.local_maps, return_reconstruction=return_map_reconstruction)
        agent_tokens, entity, history, candidates = self.agent_encoder(batch)
        b, n, a, d = candidates.shape
        candidate_valid = batch.agent_valid[:, :, None].expand(-1, -1, a).reshape(b, n * a)
        target_anchor_valid = (batch.agent_valid[:, :, None] & batch.candidate_in_view).reshape(b, n * a)
        conditioned_candidates = self.candidate_map_attention(
            candidates.reshape(b, n * a, d),
            map_tokens,
            batch.candidate_target_xy.reshape(b, n * a, 2),
            candidate_valid,
            target_anchor_valid,
        ).reshape(b, n, a, d)
        agent_tokens = torch.cat((entity[:, :, None, :], history, conditioned_candidates), dim=2)

        relation = self.relation_builder(entity, batch)
        events = self.event_encoder(batch)
        scene = self.scene_encoder(batch)
        tokens, padding = self._assemble(map_tokens, agent_tokens, relation, events, scene, batch)
        for block in self.coordination_blocks:
            tokens = block(tokens, self.token_type_ids.to(tokens.device), padding)
        tokens = self.output_norm(tokens)

        map_end = self.config.map_tokens
        agent_end = map_end + self.config.agent_tokens
        relation_end = agent_end + self.config.relation_tokens
        event_end = relation_end + self.config.event_tokens
        agent_final = tokens[:, map_end:agent_end].reshape(b, self.config.max_agents, self.config.agent_tokens_per_agent, d)
        relation_final = tokens[:, agent_end:relation_end]
        scene_final = tokens[:, event_end]
        act_final = tokens[:, event_end + 1]
        candidate_final = agent_final[:, :, 3 : 3 + self.config.num_actions]
        all_agent_logits = self.candidate_score_head(candidate_final).squeeze(-1)
        all_agent_logits = all_agent_logits.masked_fill((~batch.agent_valid)[..., None], -1.0e4)
        ego_request_delta = self.act_head(act_final)
        ego_logits = all_agent_logits[:, 0] + ego_request_delta
        return PolicyOutput(
            all_agent_logits=all_agent_logits,
            ego_logits=ego_logits,
            ego_request_delta=ego_request_delta,
            reason_logits=self.reason_head(agent_final[:, :, 0]) if self.reason_head else None,
            conflict_logits=self.conflict_head(relation_final) if self.conflict_head else None,
            scene_risk_logits=self.scene_risk_head(scene_final) if self.scene_risk_head else None,
            map_reconstruction_logits=reconstruction,
            relation_index=relation.pair_index,
            relation_valid=relation.valid,
            relation_labels=relation.labels,
            scene_embedding=scene_final,
            token_padding_mask=padding,
            final_tokens=tokens if return_tokens else None,
        )
