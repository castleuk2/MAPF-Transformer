from __future__ import annotations

import torch
from torch import nn

from .attention import PreNormAttentionBlock
from .config import ModelConfig
from .constants import ACTION_FIELD_IDS, Action, TokenField
from .map_encoder import StructuredPatchMapEncoder
from .spatial_fusion import SpatiallyRoutedCandidateMapFusion
from .structure_bias import StructuredAttentionBias
from .tokenizers import CurrentAgentTokenizer, HistoryTokenizer
from .types import PolicyBatch, PolicyOutput, PreparedPolicyContext, SemanticReconstruction


class SemanticSpatiotemporalPolicy(nn.Module):
    """Revised 256-token MAPF preference model.

    Layout:
      0..24      : Structured map patches
      25..136    : 14 current agents x 8 semantic fields
      137..248   : 4 lags x 7 tracks x 4 factual history fields
      249        : self-message query
      250..255   : up to six neighbor messages

    The baseline uses ``coordination_mode='none'`` and masks the final seven
    positions. Communication is an optional wrapper, not a requirement of the
    token design.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.map_encoder = StructuredPatchMapEncoder(config)
        self.current_tokenizer = CurrentAgentTokenizer(config)
        self.history_tokenizer = HistoryTokenizer(config)
        self.map_fusion = SpatiallyRoutedCandidateMapFusion(config)

        field_ids = self._build_field_ids()
        self.register_buffer("field_ids", field_ids, persistent=False)
        self.field_embedding = nn.Embedding(int(TokenField.COUNT), d)
        # Explicit learned absolute sequence position for every fixed slot.
        # This is distinct from semantic field identity and coordinate features.
        self.position_embedding = nn.Embedding(config.total_tokens, d)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.input_norm = nn.LayerNorm(d)
        self.structure_bias = StructuredAttentionBias(config, field_ids)
        self.blocks = nn.ModuleList(
            [
                PreNormAttentionBlock(d, config.n_heads, config.mlp_ratio, config.dropout)
                for _ in range(config.transformer_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d)

        self.message_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.message_query, std=0.02)
        self.message_slot_embedding = nn.Embedding(config.message_neighbors, d)
        self.message_source_slot_embedding = nn.Embedding(config.max_current_agents + 1, d)
        self.message_norm = nn.LayerNorm(d)

        self.candidate_score = nn.Linear(d, 1)

        if config.enable_semantic_reconstruction:
            self.current_position_x_head = nn.Linear(d, config.current_coord_vocab)
            self.current_position_y_head = nn.Linear(d, config.current_coord_vocab)
            self.current_goal_x_head = nn.Linear(d, config.goal_delta_vocab)
            self.current_goal_y_head = nn.Linear(d, config.goal_delta_vocab)
            self.current_hops_head = nn.Linear(d, config.hops_vocab)
            self.history_position_x_head = nn.Linear(d, config.history_coord_vocab)
            self.history_position_y_head = nn.Linear(d, config.history_coord_vocab)
            self.history_goal_x_head = nn.Linear(d, config.goal_delta_vocab)
            self.history_goal_y_head = nn.Linear(d, config.goal_delta_vocab)
            self.history_hops_head = nn.Linear(d, config.hops_vocab)
            self.history_selected_head = nn.Linear(d, len(Action))
            self.history_observed_head = nn.Linear(d, len(Action))
        else:
            self.current_position_x_head = None

    def _build_field_ids(self) -> torch.Tensor:
        cfg = self.config
        fields: list[int] = [int(TokenField.MAP)] * cfg.map_tokens
        current_fields = [
            int(TokenField.CURRENT_POSITION),
            int(TokenField.CURRENT_GOAL),
            int(TokenField.CURRENT_HOPS),
            *ACTION_FIELD_IDS,
        ]
        for _ in range(cfg.max_current_agents):
            fields.extend(current_fields)
        history_fields = [
            int(TokenField.HISTORY_POSITION),
            int(TokenField.HISTORY_GOAL),
            int(TokenField.HISTORY_HOPS),
            int(TokenField.HISTORY_ACTION_OUTCOME),
        ]
        for _lag in range(cfg.history_steps):
            for _track in range(cfg.history_tracks):
                fields.extend(history_fields)
        fields.append(int(TokenField.MESSAGE_QUERY))
        fields.extend([int(TokenField.NEIGHBOR_MESSAGE)] * cfg.message_neighbors)
        result = torch.tensor(fields, dtype=torch.long)
        if result.numel() != cfg.total_tokens:
            raise RuntimeError(f"field layout produced {result.numel()} tokens")
        return result

    def _coordination_tokens(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        coordination_mode: str,
        neighbor_messages: torch.Tensor | None,
        neighbor_message_valid: torch.Tensor | None,
        neighbor_message_source_slot: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if coordination_mode not in {"none", "query_only", "messages"}:
            raise ValueError("coordination_mode must be none, query_only, or messages")
        tokens = torch.zeros(
            batch_size, cfg.coordination_tokens, cfg.d_model, device=device, dtype=dtype
        )
        padding = torch.ones(
            batch_size, cfg.coordination_tokens, device=device, dtype=torch.bool
        )
        if coordination_mode == "none":
            return tokens, padding

        tokens[:, 0] = self.message_query.to(dtype=dtype)
        padding[:, 0] = False
        if coordination_mode == "query_only":
            return tokens, padding

        if neighbor_messages is None or neighbor_message_valid is None:
            raise ValueError("messages mode requires neighbor_messages and neighbor_message_valid")
        expected = (batch_size, cfg.message_neighbors, cfg.d_model)
        if tuple(neighbor_messages.shape) != expected:
            raise ValueError(f"neighbor_messages must have shape {expected}")
        if tuple(neighbor_message_valid.shape) != (batch_size, cfg.message_neighbors):
            raise ValueError("neighbor_message_valid has the wrong shape")
        if neighbor_message_source_slot is None:
            neighbor_message_source_slot = torch.full(
                (batch_size, cfg.message_neighbors),
                cfg.max_current_agents,
                device=device,
                dtype=torch.long,
            )
        source_slot = neighbor_message_source_slot.clamp(0, cfg.max_current_agents)
        slot_ids = torch.arange(cfg.message_neighbors, device=device)
        message = (
            neighbor_messages.to(dtype=dtype)
            + self.message_slot_embedding(slot_ids)[None]
            + self.message_source_slot_embedding(source_slot)
        )
        tokens[:, 1:] = self.message_norm(message)
        padding[:, 1:] = ~neighbor_message_valid.bool()
        tokens[:, 1:] = tokens[:, 1:].masked_fill(padding[:, 1:, None], 0.0)
        return tokens, padding

    def _assemble(
        self,
        map_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        batch: PolicyBatch,
        *,
        coordination_mode: str,
        neighbor_messages: torch.Tensor | None,
        neighbor_message_valid: torch.Tensor | None,
        neighbor_message_source_slot: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        b, _, d = map_tokens.shape
        current_flat = current_tokens.reshape(b, cfg.current_tokens, d)
        # [B,H,T,F,D] -> [B,T,H,F,D], matching the fixed context layout.
        history_flat = history_tokens.permute(0, 2, 1, 3, 4).reshape(
            b, cfg.history_tokens, d
        )
        coordination, coordination_padding = self._coordination_tokens(
            b,
            map_tokens.device,
            map_tokens.dtype,
            coordination_mode=coordination_mode,
            neighbor_messages=neighbor_messages,
            neighbor_message_valid=neighbor_message_valid,
            neighbor_message_source_slot=neighbor_message_source_slot,
        )
        tokens = torch.cat((map_tokens, current_flat, history_flat, coordination), dim=1)

        map_padding = torch.zeros(b, cfg.map_tokens, device=tokens.device, dtype=torch.bool)
        current_padding = (~batch.current_valid)[:, :, None].expand(
            -1, -1, cfg.current_tokens_per_agent
        ).reshape(b, cfg.current_tokens)
        history_padding = (~batch.history_valid).permute(0, 2, 1)[:, :, :, None].expand(
            -1, -1, -1, cfg.history_tokens_per_step
        ).reshape(b, cfg.history_tokens)
        padding = torch.cat(
            (map_padding, current_padding, history_padding, coordination_padding), dim=1
        )
        if tokens.shape[1] != cfg.total_tokens or padding.shape[1] != cfg.total_tokens:
            raise RuntimeError("assembled token layout does not equal 256")
        field = self.field_embedding(self.field_ids.to(tokens.device))[None]
        positions = self.position_embedding(
            torch.arange(cfg.total_tokens, device=tokens.device)
        )[None]
        tokens = self.input_norm(tokens + field + positions)
        return tokens.masked_fill(padding[..., None], 0.0), padding

    def forward(
        self,
        batch: PolicyBatch,
        *,
        coordination_mode: str = "none",
        neighbor_messages: torch.Tensor | None = None,
        neighbor_message_valid: torch.Tensor | None = None,
        neighbor_message_source_slot: torch.Tensor | None = None,
        return_tokens: bool = False,
        return_reconstruction: bool | None = None,
    ) -> PolicyOutput:
        prepared = self.prepare_context(
            batch,
            return_reconstruction=return_reconstruction,
        )
        return self.forward_prepared(
            prepared,
            coordination_mode=coordination_mode,
            neighbor_messages=neighbor_messages,
            neighbor_message_valid=neighbor_message_valid,
            neighbor_message_source_slot=neighbor_message_source_slot,
            return_tokens=return_tokens,
        )

    def prepare_context(
        self,
        batch: PolicyBatch,
        *,
        return_reconstruction: bool | None = None,
    ) -> PreparedPolicyContext:
        """Compute all tensors that do not change between communication rounds."""
        cfg = self.config
        if return_reconstruction is None:
            return_reconstruction = self.training
        map_tokens, map_reconstruction = self.map_encoder(
            batch.local_maps,
            return_reconstruction=bool(return_reconstruction and cfg.enable_map_reconstruction),
        )
        current_tokens, raw_candidates = self.current_tokenizer(batch)
        conditioned_candidates = self.map_fusion(
            raw_candidates,
            map_tokens,
            batch.current_xy,
            batch.candidate_target_core_xy,
            batch.candidate_in_core,
            batch.current_valid,
        )
        current_tokens = torch.cat(
            (current_tokens[:, :, :3], conditioned_candidates), dim=2
        )
        history_tokens = self.history_tokenizer(batch)
        # Assemble once with masked coordination slots, then retain only the
        # invariant first 249 tokens. LayerNorm is token-wise, so coordination
        # tokens can be normalized independently in each later round.
        tokens, padding = self._assemble(
            map_tokens,
            current_tokens,
            history_tokens,
            batch,
            coordination_mode="none",
            neighbor_messages=None,
            neighbor_message_valid=None,
            neighbor_message_source_slot=None,
        )
        attn_bias = (
            self.structure_bias(batch, dtype=tokens.dtype)
            if cfg.use_structure_bias
            else None
        )
        return PreparedPolicyContext(
            batch=batch,
            static_tokens=tokens[:, : cfg.coordination_offset],
            static_padding=padding[:, : cfg.coordination_offset],
            attention_bias=attn_bias,
            map_reconstruction_logits=map_reconstruction,
        )

    def forward_prepared(
        self,
        prepared: PreparedPolicyContext,
        *,
        coordination_mode: str,
        neighbor_messages: torch.Tensor | None = None,
        neighbor_message_valid: torch.Tensor | None = None,
        neighbor_message_source_slot: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> PolicyOutput:
        cfg = self.config
        batch = prepared.batch
        coordination, coordination_padding = self._coordination_tokens(
            batch.batch_size,
            prepared.static_tokens.device,
            prepared.static_tokens.dtype,
            coordination_mode=coordination_mode,
            neighbor_messages=neighbor_messages,
            neighbor_message_valid=neighbor_message_valid,
            neighbor_message_source_slot=neighbor_message_source_slot,
        )
        coordination_field = self.field_embedding(
            self.field_ids[cfg.coordination_offset :].to(coordination.device)
        )[None]
        coordination_position = self.position_embedding(
            torch.arange(
                cfg.coordination_offset,
                cfg.total_tokens,
                device=coordination.device,
            )
        )[None]
        coordination = self.input_norm(
            coordination + coordination_field + coordination_position
        )
        coordination = coordination.masked_fill(coordination_padding[..., None], 0.0)
        tokens = torch.cat((prepared.static_tokens, coordination), dim=1)
        padding = torch.cat((prepared.static_padding, coordination_padding), dim=1)
        attn_bias = prepared.attention_bias
        for block in self.blocks:
            tokens = block(tokens, attn_bias=attn_bias, key_padding_mask=padding)
        tokens = self.output_norm(tokens)

        current_final = tokens[:, cfg.current_offset : cfg.history_offset].reshape(
            batch.batch_size,
            cfg.max_current_agents,
            cfg.current_tokens_per_agent,
            cfg.d_model,
        )
        history_final = tokens[:, cfg.history_offset : cfg.coordination_offset].reshape(
            batch.batch_size,
            cfg.history_steps,
            cfg.history_tracks,
            cfg.history_tokens_per_step,
            cfg.d_model,
        )
        candidate_final = current_final[:, :, 3:]
        logits = self.candidate_score(candidate_final).squeeze(-1)
        action_valid = batch.current_valid[:, :, None] & batch.candidate_static_free
        logits = logits.masked_fill(~action_valid, -1.0e4)
        ego_logits = logits[:, 0]

        self_message = None
        if coordination_mode != "none":
            self_message = tokens[:, cfg.coordination_offset]

        semantic = None
        if cfg.enable_semantic_reconstruction:
            history_pos = history_final[:, :, :, 0]
            history_goal = history_final[:, :, :, 1]
            history_hops = history_final[:, :, :, 2]
            history_ao = history_final[:, :, :, 3]
            semantic = SemanticReconstruction(
                current_position_x=self.current_position_x_head(current_final[:, :, 0]),
                current_position_y=self.current_position_y_head(current_final[:, :, 0]),
                current_goal_x=self.current_goal_x_head(current_final[:, :, 1]),
                current_goal_y=self.current_goal_y_head(current_final[:, :, 1]),
                current_hops=self.current_hops_head(current_final[:, :, 2]),
                history_position_x=self.history_position_x_head(history_pos),
                history_position_y=self.history_position_y_head(history_pos),
                history_goal_x=self.history_goal_x_head(history_goal),
                history_goal_y=self.history_goal_y_head(history_goal),
                history_hops=self.history_hops_head(history_hops),
                history_selected_action=self.history_selected_head(history_ao),
                history_observed_move=self.history_observed_head(history_ao),
            )

        return PolicyOutput(
            ego_logits=ego_logits,
            all_current_logits=logits,
            self_message=self_message,
            map_reconstruction_logits=prepared.map_reconstruction_logits,
            semantic_reconstruction=semantic,
            token_padding_mask=padding,
            final_tokens=tokens if return_tokens else None,
        )
