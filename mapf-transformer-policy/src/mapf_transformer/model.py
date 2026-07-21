from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as nnf

from .config import ModelConfig
from .geometry import ONE_HOP_CTG_STATES


@dataclass(slots=True)
class MAPFTransformerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    action_loss: torch.Tensor | None = None
    map_reconstruction_loss: torch.Tensor | None = None
    map_reconstruction_logits: torch.Tensor | None = None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ratio: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.mlp_ratio, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.ff(self.norm2(x))


class SpatialMapEncoder(nn.Module):
    """Compresses 225 1x1 cell tokens into learned map latent tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.cell_state = nn.Embedding(config.cell_states, d)
        self.row_position = nn.Embedding(config.map_size, d)
        self.col_position = nn.Embedding(config.map_size, d)
        self.map_modality = nn.Parameter(torch.zeros(1, 1, d))
        self.cell_norm = nn.LayerNorm(d)
        self.learned_queries = nn.Parameter(torch.empty(config.map_latents, d))
        nn.init.normal_(self.learned_queries, std=0.02)
        self.query_norm = nn.LayerNorm(d)
        self.cell_norm_for_attention = nn.LayerNorm(d)
        self.cell_to_latent = nn.MultiheadAttention(
            d,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.latent_blocks = nn.ModuleList(
            [LatentBlock(config) for _ in range(config.spatial_latent_layers)]
        )
        self.output_norm = nn.LayerNorm(d)

        self.reconstruction_query = nn.Parameter(torch.empty(config.map_size * config.map_size, d))
        nn.init.normal_(self.reconstruction_query, std=0.02)
        self.reconstruction_attention = nn.MultiheadAttention(
            d,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.reconstruction_head = nn.Linear(d, config.cell_states)

        rows, cols = torch.meshgrid(
            torch.arange(config.map_size),
            torch.arange(config.map_size),
            indexing="ij",
        )
        self.register_buffer("row_ids", rows.reshape(-1), persistent=False)
        self.register_buffer("col_ids", cols.reshape(-1), persistent=False)

    def cell_embeddings(self, local_maps: torch.Tensor) -> torch.Tensor:
        if local_maps.ndim != 3:
            raise ValueError("local_maps must have shape [N,H,W]")
        if tuple(local_maps.shape[-2:]) != (self.config.map_size, self.config.map_size):
            raise ValueError("local_maps has an unexpected spatial shape")
        maps = local_maps.long().clamp(0, self.config.cell_states - 1)
        n = maps.shape[0]
        flattened = maps.reshape(n, -1)
        cell = self.cell_state(flattened)
        position = self.row_position(self.row_ids) + self.col_position(self.col_ids)
        return self.cell_norm(cell + position.unsqueeze(0) + self.map_modality)

    def forward(
        self,
        local_maps: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cells = self.cell_embeddings(local_maps)
        queries = self.learned_queries.unsqueeze(0).expand(cells.shape[0], -1, -1)
        attended, _ = self.cell_to_latent(
            self.query_norm(queries),
            self.cell_norm_for_attention(cells),
            self.cell_norm_for_attention(cells),
            need_weights=False,
        )
        latents = queries + attended
        for block in self.latent_blocks:
            latents = block(latents)
        latents = self.output_norm(latents)

        reconstruction_logits = None
        if return_reconstruction:
            reconstruction_queries = self.reconstruction_query.unsqueeze(0).expand(
                cells.shape[0], -1, -1
            )
            reconstructed, _ = self.reconstruction_attention(
                reconstruction_queries,
                latents,
                latents,
                need_weights=False,
            )
            reconstruction_logits = self.reconstruction_head(reconstructed)
        return latents, reconstruction_logits


class AgentTokenizer(nn.Module):
    """Embeds the 18-bit physical payload and non-payload metadata."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.x_embedding = nn.Embedding(16, d)
        self.y_embedding = nn.Embedding(16, d)
        self.direction_embedding = nn.Parameter(torch.empty(4, d))
        nn.init.normal_(self.direction_embedding, std=0.02)
        self.flexibility_embedding = nn.Embedding(5, d)
        self.distance_embedding = nn.Embedding(config.distance_buckets, d)
        self.one_hop_ctg_embeddings = (
            nn.ModuleList(
                [nn.Embedding(ONE_HOP_CTG_STATES, d) for _ in range(config.num_actions)]
            )
            if config.one_hop_ctg
            else None
        )

        # Metadata embeddings do not consume payload bits.
        self.role_embedding = nn.Embedding(2, d)  # neighbor / ego
        self.slot_embedding = nn.Embedding(config.agents_per_frame, d)
        self.validity_embedding = nn.Embedding(2, d)
        self.track_reset_embedding = nn.Embedding(2, d)
        self.agent_modality = nn.Parameter(torch.zeros(1, 1, d))
        self.pre_projection_norm = nn.LayerNorm(d)
        self.projection = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
        )
        self.output_norm = nn.LayerNorm(d)

        roles = torch.zeros(config.agents_per_frame, dtype=torch.long)
        roles[config.ego_slot] = 1
        self.register_buffer("role_ids", roles, persistent=False)
        self.register_buffer(
            "slot_ids", torch.arange(config.agents_per_frame, dtype=torch.long), persistent=False
        )

    def component_embeddings(
        self,
        agent_x: torch.Tensor,
        agent_y: torch.Tensor,
        action_mask: torch.Tensor,
        distance: torch.Tensor,
        one_hop_ctg: torch.Tensor | None,
        valid: torch.Tensor,
        track_reset: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if agent_x.ndim != 2:
            raise ValueError("agent fields must have shape [N, agents_per_frame]")
        n, agents = agent_x.shape
        if agents != self.config.agents_per_frame:
            raise ValueError("Unexpected number of agent slots")

        x = self.x_embedding(agent_x.long().clamp(0, 15))
        y = self.y_embedding(agent_y.long().clamp(0, 15))
        mask = action_mask.float().clamp(0, 1)
        direction_sum = torch.einsum("nad,df->naf", mask, self.direction_embedding)
        direction_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        direction = direction_sum / direction_count
        flexibility = self.flexibility_embedding(mask.sum(dim=-1).long().clamp(0, 4))
        distance_emb = self.distance_embedding(distance.long().clamp(0, 63))

        ctg = torch.zeros_like(x)
        if self.one_hop_ctg_embeddings is not None:
            if one_hop_ctg is None:
                raise ValueError("one_hop_ctg is required when config.one_hop_ctg is enabled")
            if one_hop_ctg.shape != (n, agents, self.config.num_actions):
                raise ValueError("one_hop_ctg must have shape [N, agents_per_frame, 5]")
            ctg = sum(
                embedding(one_hop_ctg[..., action].long().clamp(0, ONE_HOP_CTG_STATES - 1))
                for action, embedding in enumerate(self.one_hop_ctg_embeddings)
            ) / self.config.num_actions

        role = self.role_embedding(self.role_ids).unsqueeze(0).expand(n, -1, -1)
        slot = self.slot_embedding(self.slot_ids).unsqueeze(0).expand(n, -1, -1)
        validity = self.validity_embedding(valid.long())
        reset = self.track_reset_embedding(track_reset.long())

        return {
            "x": x, "y": y, "direction": direction, "flexibility": flexibility,
            "distance": distance_emb, "ctg": ctg, "role": role, "slot": slot,
            "validity": validity, "reset": reset,
        }

    def forward(
        self,
        agent_x: torch.Tensor,
        agent_y: torch.Tensor,
        action_mask: torch.Tensor,
        distance: torch.Tensor,
        one_hop_ctg: torch.Tensor | None,
        valid: torch.Tensor,
        track_reset: torch.Tensor,
    ) -> torch.Tensor:
        components = self.component_embeddings(
            agent_x, agent_y, action_mask, distance, one_hop_ctg, valid, track_reset
        )
        state = sum(components.values())
        projected = self.projection(self.pre_projection_norm(state + self.agent_modality))
        token = self.output_norm(state + projected)
        # Empty slots retain metadata (slot/time), but their physical content is suppressed later by validity.
        return token


class GroupedAgentTokenizer(AgentTokenizer):
    """Preserves geometry, navigation and tracking information before gated fusion."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        d = config.d_model
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.navigation_projection = nn.Sequential(
            nn.LayerNorm(4 * d), nn.Linear(4 * d, d), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.tracking_projection = nn.Sequential(
            nn.LayerNorm(4 * d), nn.Linear(4 * d, d), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.group_gate = nn.Linear(3 * d, 3)
        self.group_projection = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(d, d),
        )

    def forward(
        self,
        agent_x: torch.Tensor,
        agent_y: torch.Tensor,
        action_mask: torch.Tensor,
        distance: torch.Tensor,
        one_hop_ctg: torch.Tensor | None,
        valid: torch.Tensor,
        track_reset: torch.Tensor,
    ) -> torch.Tensor:
        c = self.component_embeddings(
            agent_x, agent_y, action_mask, distance, one_hop_ctg, valid, track_reset
        )
        geometry = self.geometry_projection(torch.cat([c["x"], c["y"]], dim=-1))
        navigation = self.navigation_projection(torch.cat(
            [c["direction"], c["flexibility"], c["distance"], c["ctg"]], dim=-1
        ))
        tracking = self.tracking_projection(torch.cat(
            [c["role"], c["slot"], c["validity"], c["reset"]], dim=-1
        ))
        groups = torch.stack([geometry, navigation, tracking], dim=-2)
        gates = torch.softmax(
            self.group_gate(torch.cat([geometry, navigation, tracking], dim=-1)), dim=-1
        )
        state = (groups * gates.unsqueeze(-1)).sum(dim=-2) + self.agent_modality
        return self.output_norm(state + self.group_projection(state))


class AgentMapFusion(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.query_norm = nn.LayerNorm(d)
        self.map_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(
            d,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(d)
        self.ff = FeedForward(d, config.mlp_ratio, config.dropout)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, agent_tokens: torch.Tensor, map_latents: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(agent_tokens),
            self.map_norm(map_latents),
            self.map_norm(map_latents),
            need_weights=False,
        )
        x = agent_tokens + attended
        x = x + self.ff(self.ff_norm(x))
        return self.output_norm(x)


class EdgeAwareGraphBlock(nn.Module):
    """Within-frame agent graph attention with geometric and conflict edge bias."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d, h = config.d_model, config.n_heads
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.output = nn.Linear(d, d)
        self.dropout = nn.Dropout(config.dropout)
        self.relative_x = nn.Embedding(29, d)
        self.relative_y = nn.Embedding(29, d)
        self.manhattan = nn.Embedding(29, d)
        self.vertex_conflict = nn.Embedding(2, d)
        self.swap_conflict = nn.Embedding(2, d)
        self.edge_norm = nn.LayerNorm(d)
        self.edge_bias = nn.Linear(d, h, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.ff = FeedForward(d, config.mlp_ratio, config.dropout)
        self.register_buffer(
            "moves", torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]]),
            persistent=False,
        )

    def _edge_embeddings(
        self,
        agent_x: torch.Tensor,
        agent_y: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.stack([agent_x, agent_y], dim=-1).long()
        relative = positions.unsqueeze(1) - positions.unsqueeze(2)  # source j - query i
        dx = relative[..., 0].clamp(-14, 14) + 14
        dy = relative[..., 1].clamp(-14, 14) + 14
        distance = relative.abs().sum(dim=-1).clamp(0, 28)

        candidates = positions.unsqueeze(2) + self.moves.view(1, 1, 4, 2)
        allowed = action_mask.bool()
        same_destination = (
            candidates.unsqueeze(2).unsqueeze(4)
            == candidates.unsqueeze(1).unsqueeze(3)
        ).all(dim=-1)
        allowed_pairs = allowed.unsqueeze(2).unsqueeze(4) & allowed.unsqueeze(1).unsqueeze(3)
        vertex = (same_destination & allowed_pairs).any(dim=(-1, -2)).long()

        candidate_i_is_j = (
            candidates.unsqueeze(2) == positions.unsqueeze(1).unsqueeze(3)
        ).all(dim=-1)
        candidate_j_is_i = (
            candidates.unsqueeze(1) == positions.unsqueeze(2).unsqueeze(3)
        ).all(dim=-1)
        swap = (
            (candidate_i_is_j & allowed.unsqueeze(2)).any(dim=-1)
            & (candidate_j_is_i & allowed.unsqueeze(1)).any(dim=-1)
        ).long()
        return self.edge_norm(
            self.relative_x(dx) + self.relative_y(dy) + self.manhattan(distance)
            + self.vertex_conflict(vertex) + self.swap_conflict(swap)
        )

    def forward(
        self,
        x: torch.Tensor,
        agent_x: torch.Tensor,
        agent_y: torch.Tensor,
        action_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, agents, d = x.shape
        normalized = self.norm1(x)
        qkv = self.qkv(normalized).reshape(batch, agents, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale
        edges = self._edge_embeddings(agent_x, agent_y, action_mask)
        logits = logits + self.edge_bias(edges).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~valid[:, None, None, :].bool(), float("-inf"))
        logits = torch.where(valid[:, None, :, None].bool(), logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid[:, None, :, None] * valid[:, None, None, :]
        attended = torch.einsum("bhij,bjhd->bihd", weights, v).reshape(batch, agents, d)
        x = x + self.dropout(self.output(attended))
        x = x + self.ff(self.norm2(x))
        return x * valid.unsqueeze(-1).to(x.dtype)


class TransitionTokenizer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.previous_action = nn.Embedding(6, d)  # five actions + START
        self.actual_move = nn.Embedding(5, d)
        self.outcome = nn.Embedding(4, d)
        self.visible_count = nn.Embedding(config.agents_per_frame + 1, d)
        self.transition_modality = nn.Parameter(torch.zeros(1, d))
        self.norm = nn.LayerNorm(d)
        self.projection = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(
        self,
        previous_action: torch.Tensor,
        actual_move: torch.Tensor,
        outcome: torch.Tensor,
        visible_count: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.previous_action(previous_action.long().clamp(0, 5))
            + self.actual_move(actual_move.long().clamp(0, 4))
            + self.outcome(outcome.long().clamp(0, 3))
            + self.visible_count(visible_count.long().clamp(0, self.visible_count.num_embeddings - 1))
            + self.transition_modality
        )
        x = x + self.projection(self.norm(x))
        return self.norm(x)


class TemporalBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.mlp_ratio, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + self.dropout(attended)
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(~token_valid.unsqueeze(-1), 0.0)


class MAPFTransformer(nn.Module):
    """Hierarchical spatial-map + temporal policy Transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.map_encoder = SpatialMapEncoder(config)
        self.agent_tokenizer = (
            GroupedAgentTokenizer(config)
            if config.metadata_encoder == "grouped"
            else AgentTokenizer(config)
        )
        self.agent_map_fusion = AgentMapFusion(config)
        self.graph_blocks = nn.ModuleList(
            [EdgeAwareGraphBlock(config) for _ in range(config.graph_layers)]
        )
        self.transition_tokenizer = TransitionTokenizer(config)

        self.frame_position = nn.Embedding(config.history_frames, config.d_model)
        self.within_frame_position = nn.Embedding(config.tokens_per_frame, config.d_model)
        self.temporal_modality = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
        self.act_query = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.act_query, std=0.02)
        self.act_modality = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.temporal_blocks = nn.ModuleList(
            [TemporalBlock(config) for _ in range(config.temporal_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.action_head = nn.Linear(config.d_model, config.num_actions)

        frame_ids = torch.arange(config.history_frames).view(-1, 1).expand(
            -1, config.tokens_per_frame
        )
        within_ids = torch.arange(config.tokens_per_frame).view(1, -1).expand(
            config.history_frames, -1
        )
        self.register_buffer("frame_ids", frame_ids.reshape(-1), persistent=False)
        self.register_buffer("within_ids", within_ids.reshape(-1), persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_maps(
        self,
        local_maps: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.map_encoder(local_maps, return_reconstruction=return_reconstruction)

    def encode_frame_from_latents(
        self,
        map_latents: torch.Tensor,
        agent_x: torch.Tensor,
        agent_y: torch.Tensor,
        action_mask: torch.Tensor,
        distance: torch.Tensor,
        one_hop_ctg: torch.Tensor | None,
        agent_valid: torch.Tensor,
        track_reset: torch.Tensor,
        previous_action: torch.Tensor,
        actual_move: torch.Tensor,
        outcome: torch.Tensor,
        visible_count: torch.Tensor,
    ) -> torch.Tensor:
        agents = self.agent_tokenizer(
            agent_x,
            agent_y,
            action_mask,
            distance,
            one_hop_ctg,
            agent_valid,
            track_reset,
        )
        conditioned_agents = self.agent_map_fusion(agents, map_latents)
        for graph_block in self.graph_blocks:
            conditioned_agents = graph_block(
                conditioned_agents,
                agent_x,
                agent_y,
                action_mask,
                agent_valid,
            )
        # Suppress invalid physical slots after metadata/fusion while retaining stable shape.
        conditioned_agents = conditioned_agents * agent_valid.unsqueeze(-1).to(conditioned_agents.dtype)
        transition = self.transition_tokenizer(
            previous_action,
            actual_move,
            outcome,
            visible_count,
        ).unsqueeze(1)
        return torch.cat([conditioned_agents, transition], dim=1)

    def encode_frames(
        self,
        batch: Mapping[str, torch.Tensor],
        return_reconstruction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        local_maps = batch["local_maps"]
        if local_maps.ndim != 4:
            raise ValueError("local_maps must have shape [B,F,H,W]")
        b, f = local_maps.shape[:2]
        if f != self.config.history_frames:
            raise ValueError("Unexpected history length")
        flattened_maps = local_maps.reshape(b * f, self.config.map_size, self.config.map_size)
        map_latents, reconstruction = self.encode_maps(
            flattened_maps,
            return_reconstruction=return_reconstruction,
        )

        def flatten_agent(name: str) -> torch.Tensor:
            tensor = batch[name]
            return tensor.reshape(b * f, *tensor.shape[2:])

        frame_tokens = self.encode_frame_from_latents(
            map_latents=map_latents,
            agent_x=flatten_agent("agent_x"),
            agent_y=flatten_agent("agent_y"),
            action_mask=flatten_agent("action_mask"),
            distance=flatten_agent("distance"),
            one_hop_ctg=(
                flatten_agent("one_hop_ctg") if self.config.one_hop_ctg else None
            ),
            agent_valid=flatten_agent("agent_valid"),
            track_reset=flatten_agent("track_reset"),
            previous_action=batch["previous_action"].reshape(b * f),
            actual_move=batch["actual_move"].reshape(b * f),
            outcome=batch["outcome"].reshape(b * f),
            visible_count=batch["visible_count"].reshape(b * f),
        )
        frame_tokens = frame_tokens.reshape(
            b,
            f,
            self.config.tokens_per_frame,
            self.config.d_model,
        )
        if reconstruction is not None:
            reconstruction = reconstruction.reshape(
                b,
                f,
                self.config.map_size * self.config.map_size,
                self.config.cell_states,
            )
        return frame_tokens, reconstruction

    def _build_temporal_attention_mask(
        self,
        frame_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds block-causal mask: bidirectional within frame, causal across frames."""
        b, f = frame_valid.shape
        p = self.config.tokens_per_frame
        regular_tokens = f * p
        total_tokens = regular_tokens + 1
        device = frame_valid.device

        regular_frame_ids = torch.arange(f, device=device).repeat_interleave(p)
        query_frames = regular_frame_ids[:, None]
        key_frames = regular_frame_ids[None, :]
        regular_mask = key_frames > query_frames

        base = torch.ones((total_tokens, total_tokens), dtype=torch.bool, device=device)
        base[:regular_tokens, :regular_tokens] = regular_mask
        base[:regular_tokens, regular_tokens] = True  # ordinary tokens cannot read ACT
        base[regular_tokens, :] = False  # ACT reads all context and itself

        token_valid = torch.cat(
            [frame_valid.repeat_interleave(p, dim=1), torch.ones((b, 1), dtype=torch.bool, device=device)],
            dim=1,
        )
        masks = base.unsqueeze(0).expand(b, -1, -1).clone()
        invalid_keys = ~token_valid
        masks |= invalid_keys.unsqueeze(1).expand(-1, total_tokens, -1)

        # Invalid query rows attend only to themselves; this prevents all-masked NaNs.
        for batch_index in range(b):
            invalid_queries = torch.nonzero(~token_valid[batch_index], as_tuple=False).flatten()
            if invalid_queries.numel() > 0:
                masks[batch_index, invalid_queries, :] = True
                masks[batch_index, invalid_queries, invalid_queries] = False

        masks = masks.repeat_interleave(self.config.n_heads, dim=0)
        return masks, token_valid

    def forward_encoded_frames(
        self,
        frame_tokens: torch.Tensor,
        frame_valid: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> MAPFTransformerOutput:
        if frame_tokens.ndim != 4:
            raise ValueError("frame_tokens must have shape [B,F,P,D]")
        b, f, p, d = frame_tokens.shape
        expected = (
            self.config.history_frames,
            self.config.tokens_per_frame,
            self.config.d_model,
        )
        if (f, p, d) != expected:
            raise ValueError(f"Unexpected frame token shape: {(f, p, d)} vs {expected}")

        frame_pos = self.frame_position(self.frame_ids).reshape(1, f, p, d)
        within_pos = self.within_frame_position(self.within_ids).reshape(1, f, p, d)
        temporal = frame_tokens + frame_pos + within_pos + self.temporal_modality
        temporal = temporal.reshape(b, f * p, d)
        act = (self.act_query + self.act_modality).expand(b, -1, -1)
        x = torch.cat([temporal, act], dim=1)
        attention_mask, token_valid = self._build_temporal_attention_mask(frame_valid.bool())
        x = x.masked_fill(~token_valid.unsqueeze(-1), 0.0)
        for block in self.temporal_blocks:
            x = block(x, attention_mask, token_valid)
        act_state = self.final_norm(x[:, -1])
        logits = self.action_head(act_state)
        action_loss = nnf.cross_entropy(logits, targets.long()) if targets is not None else None
        return MAPFTransformerOutput(logits=logits, loss=action_loss, action_loss=action_loss)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        targets: torch.Tensor | None = None,
        return_reconstruction: bool | None = None,
    ) -> MAPFTransformerOutput:
        if targets is None and "target" in batch:
            targets = batch["target"]
        if return_reconstruction is None:
            return_reconstruction = self.training
        need_reconstruction = self.config.aux_map_reconstruction and return_reconstruction
        frames, reconstruction = self.encode_frames(
            batch,
            return_reconstruction=need_reconstruction,
        )
        output = self.forward_encoded_frames(frames, batch["frame_valid"], targets=targets)
        output.map_reconstruction_logits = reconstruction

        reconstruction_loss = None
        if reconstruction is not None:
            valid = batch["frame_valid"].bool()
            if valid.any():
                target_maps = batch["local_maps"].reshape(
                    batch["local_maps"].shape[0],
                    batch["local_maps"].shape[1],
                    -1,
                ).long()
                reconstruction_loss = nnf.cross_entropy(
                    reconstruction[valid].reshape(-1, self.config.cell_states),
                    target_maps[valid].reshape(-1),
                )
                if output.loss is None:
                    output.loss = self.config.aux_map_loss_weight * reconstruction_loss
                else:
                    output.loss = output.loss + self.config.aux_map_loss_weight * reconstruction_loss
        output.map_reconstruction_loss = reconstruction_loss
        return output

    @torch.no_grad()
    def predict_actions(
        self,
        batch: Mapping[str, torch.Tensor],
        sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        logits = self.forward(batch).logits
        if sample:
            temperature = max(float(temperature), 1e-6)
            probabilities = torch.softmax(logits / temperature, dim=-1)
            actions = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        else:
            actions = logits.argmax(dim=-1)
        if was_training:
            self.train()
        return actions
