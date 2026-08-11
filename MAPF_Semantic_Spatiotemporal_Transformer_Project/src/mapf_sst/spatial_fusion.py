from __future__ import annotations

import torch
from torch import nn

from .attention import MultiheadAttentionWithBias
from .config import ModelConfig


class SpatiallyRoutedCandidateMapFusion(nn.Module):
    """Deterministic source/target gather plus sparse local map attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.source_projection = nn.Linear(d, d)
        self.target_projection = nn.Linear(d, d)
        self.source_cell_embedding = nn.Embedding(9, d)
        self.target_cell_embedding = nn.Embedding(10, d)  # 9 cells + outside
        self.query_norm = nn.LayerNorm(d)
        self.map_norm = nn.LayerNorm(d)
        self.attn = MultiheadAttentionWithBias(d, config.n_heads, config.dropout)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * config.mlp_ratio, d),
            nn.Dropout(config.dropout),
        )
        self.source_bonus = nn.Parameter(torch.zeros(config.n_heads))
        self.target_bonus = nn.Parameter(torch.zeros(config.n_heads))
        nn.init.normal_(self.source_bonus, std=0.01)
        nn.init.normal_(self.target_bonus, std=0.01)

        pps = config.patches_per_side
        rows, cols = torch.meshgrid(torch.arange(pps), torch.arange(pps), indexing="ij")
        self.register_buffer(
            "patch_rc", torch.stack((rows.reshape(-1), cols.reshape(-1)), dim=-1), persistent=False
        )

    @staticmethod
    def _gather(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        b, _, d = tokens.shape
        expanded = index[..., None].expand(*index.shape, d)
        source = tokens[:, None, None, :, :].expand(-1, index.shape[1], index.shape[2], -1, -1)
        return torch.gather(source, 3, expanded[..., None, :]).squeeze(3)

    def forward(
        self,
        candidates: torch.Tensor,
        map_tokens: torch.Tensor,
        current_xy: torch.Tensor,
        target_core_xy: torch.Tensor,
        target_in_core: torch.Tensor,
        current_valid: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.use_spatial_map_fusion:
            return candidates
        cfg = self.config
        b, n, a, d = candidates.shape
        center = cfg.core_map_size // 2
        source_core = current_xy + center
        source_core = source_core.clamp(0, cfg.core_map_size - 1)
        source_patch_rc = torch.div(source_core, cfg.patch_size, rounding_mode="floor")
        target_clamped = target_core_xy.clamp(0, cfg.core_map_size - 1)
        target_patch_rc = torch.div(target_clamped, cfg.patch_size, rounding_mode="floor")
        source_index = source_patch_rc[..., 0] * cfg.patches_per_side + source_patch_rc[..., 1]
        target_index = target_patch_rc[..., 0] * cfg.patches_per_side + target_patch_rc[..., 1]

        source_index_a = source_index[:, :, None].expand(-1, -1, a)
        source_map = self._gather(map_tokens, source_index_a)
        target_map = self._gather(map_tokens, target_index)
        target_map = target_map * target_in_core[..., None].to(target_map.dtype)

        source_cell = (source_core[..., 0] % cfg.patch_size) * cfg.patch_size + (
            source_core[..., 1] % cfg.patch_size
        )
        source_cell = source_cell[:, :, None].expand(-1, -1, a)
        target_cell = (target_clamped[..., 0] % cfg.patch_size) * cfg.patch_size + (
            target_clamped[..., 1] % cfg.patch_size
        )
        target_cell = torch.where(target_in_core, target_cell, torch.full_like(target_cell, 9))

        x = (
            candidates
            + self.source_projection(source_map)
            + self.target_projection(target_map)
            + self.source_cell_embedding(source_cell)
            + self.target_cell_embedding(target_cell)
        )

        q = n * a
        source_pr = source_patch_rc[:, :, None, :].expand(-1, -1, a, -1).reshape(b, q, 2)
        target_pr = target_patch_rc.reshape(b, q, 2)
        target_ok = target_in_core.reshape(b, q)
        patch = self.patch_rc[None, None, :, :]
        radius = cfg.map_attention_radius
        source_local = (patch - source_pr[:, :, None, :]).abs().amax(dim=-1) <= radius
        target_local = (patch - target_pr[:, :, None, :]).abs().amax(dim=-1) <= radius
        allowed = source_local | (target_local & target_ok[:, :, None])
        bias = torch.zeros(b, cfg.n_heads, q, cfg.map_tokens, device=x.device, dtype=x.dtype)
        bias = bias.masked_fill(~allowed[:, None], torch.finfo(x.dtype).min)
        source_same = (patch == source_pr[:, :, None, :]).all(dim=-1)
        target_same = (patch == target_pr[:, :, None, :]).all(dim=-1) & target_ok[:, :, None]
        bias = bias + self.source_bonus[None, :, None, None] * source_same[:, None]
        bias = bias + self.target_bonus[None, :, None, None] * target_same[:, None]

        flat = x.reshape(b, q, d)
        candidate_valid = current_valid[:, :, None].expand(-1, -1, a).reshape(b, q)
        attended, _ = self.attn(
            self.query_norm(flat),
            self.map_norm(map_tokens),
            self.map_norm(map_tokens),
            attn_bias=bias,
            query_padding_mask=~candidate_valid,
        )
        flat = flat + attended
        flat = flat + self.ff(self.ff_norm(flat))
        flat = flat.masked_fill((~candidate_valid)[..., None], 0.0)
        return flat.reshape(b, n, a, d)
