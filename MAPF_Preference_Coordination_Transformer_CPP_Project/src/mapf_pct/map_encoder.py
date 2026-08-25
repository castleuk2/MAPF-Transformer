from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .attention import MultiheadAttentionWithBias, PreNormAttentionBlock
from .config import ModelConfig
from .structured_map import ModelConfig as StructuredMapConfig
from .structured_map import StructuredMapTransformer


class _LegacyStructuredPatchMapEncoder(nn.Module):
    """17x17 local map -> 25 structurally anchored 3x3 patch tokens.

    Each token receives a fixed 34-D raw feature:
      * 9 binary cell states as 18-D one-hot slots,
      * 4 directions x 3 boundary openings = 12 bits,
      * 4 outer-core edge bits.

    The token therefore keeps the same 3x3 spatial responsibility before and
    after self-attention, matching the attached Structured-25 design.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        pps = config.patches_per_side
        self.patch_mlp = nn.Sequential(
            nn.Linear(34, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.row_embedding = nn.Embedding(pps, d)
        self.col_embedding = nn.Embedding(pps, d)
        self.center_embedding = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.center_embedding, std=0.02)

        self.relative_bias = nn.Parameter(torch.zeros(config.n_heads, 2 * pps - 1, 2 * pps - 1))
        self.connectivity_bias = nn.Parameter(torch.zeros(config.n_heads, 4, 8))
        nn.init.normal_(self.relative_bias, std=0.01)
        nn.init.normal_(self.connectivity_bias, std=0.01)

        self.blocks = nn.ModuleList(
            [
                PreNormAttentionBlock(
                    d,
                    config.n_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.map_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d)
        self.decoder = nn.Linear(d, config.patch_size * config.patch_size * 2)

        rows, cols = torch.meshgrid(
            torch.arange(pps),
            torch.arange(pps),
            indexing="ij",
        )
        patch_rc = torch.stack((rows.reshape(-1), cols.reshape(-1)), dim=-1)
        self.register_buffer("patch_rc", patch_rc, persistent=False)
        dr = patch_rc[:, None, 0] - patch_rc[None, :, 0]
        dc = patch_rc[:, None, 1] - patch_rc[None, :, 1]
        self.register_buffer("relative_dr", dr + pps - 1, persistent=False)
        self.register_buffer("relative_dc", dc + pps - 1, persistent=False)

        # Direction order: North, South, West, East.
        adjacent_direction = torch.full((config.map_tokens, config.map_tokens), -1, dtype=torch.long)
        for i, (ri, ci) in enumerate(patch_rc.tolist()):
            for j, (rj, cj) in enumerate(patch_rc.tolist()):
                if rj == ri - 1 and cj == ci:
                    adjacent_direction[i, j] = 0
                elif rj == ri + 1 and cj == ci:
                    adjacent_direction[i, j] = 1
                elif rj == ri and cj == ci - 1:
                    adjacent_direction[i, j] = 2
                elif rj == ri and cj == ci + 1:
                    adjacent_direction[i, j] = 3
        self.register_buffer("adjacent_direction", adjacent_direction, persistent=False)

    def _validate_map(self, local_maps: torch.Tensor) -> None:
        if local_maps.ndim != 3:
            raise ValueError("local_maps must have shape [B,17,17]")
        expected = (self.config.local_map_size, self.config.local_map_size)
        if tuple(local_maps.shape[-2:]) != expected:
            raise ValueError(f"expected local map spatial shape {expected}, got {tuple(local_maps.shape[-2:])}")

    def extract_raw_features(self, local_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_map(local_maps)
        maps = local_maps.long().clamp(0, 1)
        b = maps.shape[0]
        p = self.config.patch_size
        pps = self.config.patches_per_side
        core = maps[:, 1:-1, 1:-1]
        patches = core.unfold(1, p, p).unfold(2, p, p).contiguous()
        # [B,5,5,3,3] -> [B,25,9]
        patch_cells = patches.view(b, pps * pps, p * p)
        cell_one_hot = F.one_hot(patch_cells, num_classes=2).float().reshape(b, -1, 18)

        openings = torch.zeros(b, pps, pps, 4, p, device=maps.device, dtype=torch.float32)
        for pr in range(pps):
            for pc in range(pps):
                r0 = 1 + pr * p
                c0 = 1 + pc * p
                # Free on both sides of the patch boundary.
                openings[:, pr, pc, 0] = ((maps[:, r0, c0 : c0 + p] == 0) & (maps[:, r0 - 1, c0 : c0 + p] == 0)).float()
                openings[:, pr, pc, 1] = ((maps[:, r0 + p - 1, c0 : c0 + p] == 0) & (maps[:, r0 + p, c0 : c0 + p] == 0)).float()
                openings[:, pr, pc, 2] = ((maps[:, r0 : r0 + p, c0] == 0) & (maps[:, r0 : r0 + p, c0 - 1] == 0)).float()
                openings[:, pr, pc, 3] = ((maps[:, r0 : r0 + p, c0 + p - 1] == 0) & (maps[:, r0 : r0 + p, c0 + p] == 0)).float()
        openings_flat = openings.view(b, pps * pps, 12)

        edge = torch.zeros(pps, pps, 4, device=maps.device, dtype=torch.float32)
        edge[0, :, 0] = 1.0
        edge[-1, :, 1] = 1.0
        edge[:, 0, 2] = 1.0
        edge[:, -1, 3] = 1.0
        edge = edge.view(1, pps * pps, 4).expand(b, -1, -1)
        raw = torch.cat((cell_one_hot, openings_flat, edge), dim=-1)
        return raw, openings

    def _attention_bias(self, openings: torch.Tensor) -> torch.Tensor:
        b = openings.shape[0]
        # [H,25,25]
        relative = self.relative_bias[:, self.relative_dr, self.relative_dc]
        bias = relative.unsqueeze(0).expand(b, -1, -1, -1).clone()

        opening_bits = openings.reshape(b, self.config.map_tokens, 4, 3).long()
        patterns = opening_bits[..., 0] * 4 + opening_bits[..., 1] * 2 + opening_bits[..., 2]
        for direction in range(4):
            adjacency = (self.adjacent_direction == direction).to(dtype=bias.dtype)
            # Table lookup for each query patch's 3-bit opening pattern.
            table = self.connectivity_bias[:, direction, :]  # [H,8]
            values = table[:, patterns[:, :, direction]]  # [H,B,25]
            values = values.permute(1, 0, 2).unsqueeze(-1)  # [B,H,25,1]
            bias = bias + values * adjacency[None, None, :, :]
        return bias

    def forward(
        self,
        local_maps: torch.Tensor,
        *,
        return_reconstruction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw, openings = self.extract_raw_features(local_maps)
        tokens = self.patch_mlp(raw)
        position = self.row_embedding(self.patch_rc[:, 0]) + self.col_embedding(self.patch_rc[:, 1])
        tokens = tokens + position.unsqueeze(0)
        center_index = self.config.map_tokens // 2
        tokens[:, center_index] = tokens[:, center_index] + self.center_embedding
        bias = self._attention_bias(openings)
        for block in self.blocks:
            tokens = block(tokens, attn_bias=bias)
        tokens = self.output_norm(tokens)

        reconstruction = None
        if return_reconstruction:
            patch_logits = self.decoder(tokens).view(
                local_maps.shape[0],
                self.config.patches_per_side,
                self.config.patches_per_side,
                self.config.patch_size,
                self.config.patch_size,
                2,
            )
            # [B,Pr,Pc,r,c,2] -> [B,Pr,r,Pc,c,2] -> [B,15,15,2]
            reconstruction = patch_logits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                local_maps.shape[0], self.config.core_map_size, self.config.core_map_size, 2
            )
        return tokens, reconstruction


class CandidateToMapCrossAttention(nn.Module):
    """Map-conditions every action candidate using its target-patch anchor."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query_norm = nn.LayerNorm(config.d_model)
        self.map_norm = nn.LayerNorm(config.d_model)
        self.attn = MultiheadAttentionWithBias(config.d_model, config.n_heads, config.dropout)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.mlp_ratio, config.d_model),
            nn.Dropout(config.dropout),
        )
        pps = config.patches_per_side
        self.relative_target_bias = nn.Parameter(
            torch.zeros(config.n_heads, 2 * pps - 1, 2 * pps - 1)
        )
        self.target_patch_bonus = nn.Parameter(torch.zeros(config.n_heads))
        nn.init.normal_(self.relative_target_bias, std=0.01)
        nn.init.normal_(self.target_patch_bonus, std=0.01)
        rows, cols = torch.meshgrid(torch.arange(pps), torch.arange(pps), indexing="ij")
        self.register_buffer(
            "patch_rc",
            torch.stack((rows.reshape(-1), cols.reshape(-1)), dim=-1),
            persistent=False,
        )

    def _bias(self, target_xy: torch.Tensor, target_valid: torch.Tensor) -> torch.Tensor:
        # target_xy: [B,Q,2] in 0..14 core coordinates.
        pps = self.config.patches_per_side
        target_patch = torch.div(target_xy.clamp(0, self.config.core_map_size - 1), self.config.patch_size, rounding_mode="floor")
        dr = target_patch[:, :, None, 0] - self.patch_rc[None, None, :, 0]
        dc = target_patch[:, :, None, 1] - self.patch_rc[None, None, :, 1]
        dr_index = (dr + pps - 1).clamp(0, 2 * pps - 2)
        dc_index = (dc + pps - 1).clamp(0, 2 * pps - 2)
        flat_index = dr_index * (2 * pps - 1) + dc_index
        table = self.relative_target_bias.view(self.config.n_heads, -1)
        bias = table[:, flat_index].permute(1, 0, 2, 3).contiguous()
        same = (dr == 0) & (dc == 0)
        bias = bias + self.target_patch_bonus[None, :, None, None] * same[:, None].to(bias.dtype)
        return bias * target_valid[:, None, :, None].to(bias.dtype)

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        target_xy: torch.Tensor,
        candidate_valid: torch.Tensor,
        target_anchor_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_anchor_valid is None:
            target_anchor_valid = candidate_valid
        bias = self._bias(target_xy, target_anchor_valid)
        attended, _ = self.attn(
            self.query_norm(candidate_tokens),
            self.map_norm(map_tokens),
            self.map_norm(map_tokens),
            attn_bias=bias,
            query_padding_mask=~candidate_valid,
        )
        x = candidate_tokens + attended
        x = x + self.ff(self.ff_norm(x))
        return x.masked_fill((~candidate_valid)[..., None], 0.0)


class StructuredPatchMapEncoder(StructuredMapTransformer):
    """Exact Structured-25 map encoder with the PCT project's tuple API.

    The implementation and parameter layout are frozen to the local
    ``mapf-structured-map-transformer`` project. Only this forward adapter is
    PCT-specific; Agent, Relation, Scene, and Coordination modules remain
    outside the map encoder.
    """

    def __init__(self, config: ModelConfig) -> None:
        structured_config = StructuredMapConfig(
            halo_size=config.local_map_size,
            core_size=config.core_map_size,
            patch_size=config.patch_size,
            num_cell_states=2,
            free_state=0,
            occupied_state=1,
            d_model=config.d_model,
            patch_hidden_dim=config.d_model,
            num_heads=config.n_heads,
            num_layers=config.map_layers,
            ffn_multiplier=config.mlp_ratio,
            dropout=config.dropout,
            position_encoding="learned",
            use_center_embedding=True,
            use_relative_position_bias=True,
            use_connectivity_bias=True,
            include_port_openings=True,
            include_outer_edge_mask=True,
            include_port_known_mask=False,
            final_norm=True,
            reconstruct_ports=False,
            reconstruction_classes=2,
        )
        super().__init__(structured_config)
        self.pct_config = config

    def extract_raw_features(self, local_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self._build_geometry(local_maps)
        batch = local_maps.shape[0] if local_maps.ndim == 3 else 1
        openings = geometry.port_open.reshape(
            batch,
            self.config.patch_grid_size,
            self.config.patch_grid_size,
            4,
            self.config.patch_size,
        )
        return geometry.features, openings

    def forward(
        self,
        local_maps: torch.Tensor,
        *,
        return_reconstruction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = super().forward(local_maps, return_attention=False)
        reconstruction = output.reconstruction_logits if return_reconstruction else None
        return output.latent_tokens, reconstruction
