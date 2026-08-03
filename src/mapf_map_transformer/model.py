from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .geometry import EAST, NORTH, SOUTH, WEST, PatchGeometry, build_patch_geometry, unpatchify
from .positional import build_position_encoding


@dataclass(slots=True)
class MapEncoderOutput:
    latent_tokens: torch.Tensor
    reconstruction_logits: torch.Tensor
    patch_logits: torch.Tensor
    port_logits: torch.Tensor | None
    patch_geometry: PatchGeometry
    attention_maps: tuple[torch.Tensor, ...] | None = None


class PatchGeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SpatialSelfAttention(nn.Module):
    """Full 25-token attention with 2D relative and patch-connectivity bias."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        grid_size: int,
        dropout: float,
        use_relative_position_bias: bool,
        use_connectivity_bias: bool,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.scale = self.head_dim**-0.5
        self.use_relative_position_bias = use_relative_position_bias
        self.use_connectivity_bias = use_connectivity_bias

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        relative_size = 2 * grid_size - 1
        if use_relative_position_bias:
            self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, relative_size * relative_size))
            relative_index = self._build_relative_index(grid_size)
            self.register_buffer("relative_position_index", relative_index, persistent=False)
        else:
            self.relative_position_bias = None
            self.register_buffer("relative_position_index", torch.empty(0, dtype=torch.long), persistent=False)

        if use_connectivity_bias:
            # [head, source direction (N,S,W,E), three-bit opening code (0..7)]
            self.connectivity_bias = nn.Parameter(torch.zeros(num_heads, 4, 8))
            self._register_neighbor_indices(grid_size)
        else:
            self.connectivity_bias = None
            for name in ("src_n", "dst_n", "src_s", "dst_s", "src_w", "dst_w", "src_e", "dst_e"):
                self.register_buffer(name, torch.empty(0, dtype=torch.long), persistent=False)

        self.reset_parameters()

    @staticmethod
    def _build_relative_index(grid_size: int) -> torch.Tensor:
        rows = torch.arange(grid_size)
        cols = torch.arange(grid_size)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([row_grid.reshape(-1), col_grid.reshape(-1)], dim=-1)
        relative = coords[:, None, :] - coords[None, :, :]
        relative += grid_size - 1
        relative_size = 2 * grid_size - 1
        return relative[..., 0] * relative_size + relative[..., 1]

    def _register_neighbor_indices(self, grid_size: int) -> None:
        index = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
        pairs = {
            "n": (index[1:, :].reshape(-1), index[:-1, :].reshape(-1)),
            "s": (index[:-1, :].reshape(-1), index[1:, :].reshape(-1)),
            "w": (index[:, 1:].reshape(-1), index[:, :-1].reshape(-1)),
            "e": (index[:, :-1].reshape(-1), index[:, 1:].reshape(-1)),
        }
        for suffix, (src, dst) in pairs.items():
            self.register_buffer(f"src_{suffix}", src, persistent=False)
            self.register_buffer(f"dst_{suffix}", dst, persistent=False)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        if self.relative_position_bias is not None:
            nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        if self.connectivity_bias is not None:
            nn.init.zeros_(self.connectivity_bias)
            # A small prior: any nonzero opening initially receives a weak positive bias.
            with torch.no_grad():
                self.connectivity_bias[:, :, 1:] = 0.05

    def _relative_bias(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        assert self.relative_position_bias is not None
        bias = self.relative_position_bias[:, self.relative_position_index.reshape(-1)]
        return bias.reshape(self.num_heads, self.num_tokens, self.num_tokens).to(device=device, dtype=dtype)

    def _connectivity_pair_bias(
        self,
        connectivity_codes: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert self.connectivity_bias is not None
        batch = connectivity_codes.shape[0]
        bias = torch.zeros(
            batch,
            self.num_heads,
            self.num_tokens,
            self.num_tokens,
            device=connectivity_codes.device,
            dtype=dtype,
        )
        direction_specs = (
            (NORTH, self.src_n, self.dst_n),
            (SOUTH, self.src_s, self.dst_s),
            (WEST, self.src_w, self.dst_w),
            (EAST, self.src_e, self.dst_e),
        )
        for direction, src, dst in direction_specs:
            codes = connectivity_codes[:, src, direction].to(torch.long)  # [B,P]
            one_hot = F.one_hot(codes, num_classes=8).to(dtype=dtype)
            table = self.connectivity_bias[:, direction, :].to(dtype=dtype)
            values = torch.einsum("bpk,hk->bhp", one_hot, table)
            bias[:, :, src, dst] = values
        return bias

    def forward(
        self,
        tokens: torch.Tensor,
        connectivity_codes: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, num_tokens, _ = tokens.shape
        if num_tokens != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {num_tokens}.")
        qkv = self.qkv(tokens).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if self.use_relative_position_bias:
            logits = logits + self._relative_bias(logits.dtype, logits.device).unsqueeze(0)
        if self.use_connectivity_bias:
            logits = logits + self._connectivity_pair_bias(connectivity_codes, dtype=logits.dtype)

        attention = torch.softmax(logits, dim=-1)
        dropped_attention = self.attention_dropout(attention)
        output = torch.matmul(dropped_attention, value)
        output = output.transpose(1, 2).reshape(batch, num_tokens, self.d_model)
        output = self.output_dropout(self.output_projection(output))
        return output, attention if return_attention else None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.network(tokens)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = SpatialSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            grid_size=config.patch_grid_size,
            dropout=config.dropout,
            use_relative_position_bias=config.use_relative_position_bias,
            use_connectivity_bias=config.use_connectivity_bias,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(
            config.d_model,
            config.d_model * config.ffn_multiplier,
            config.dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        connectivity_codes: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        update, attention = self.attention(
            self.norm1(tokens),
            connectivity_codes,
            return_attention=return_attention,
        )
        tokens = tokens + update
        tokens = tokens + self.feed_forward(self.norm2(tokens))
        return tokens, attention


class StructuredMapTransformer(nn.Module):
    """17x17 halo map -> 25 spatially anchored 3x3 map tokens.

    The central 15x15 cells are divided into a 5x5 grid of non-overlapping 3x3
    patches. The halo is never tokenized; it supplies the neighbor cell needed
    to compute true openings at the outer edge of the 15x15 core.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.patch_encoder = PatchGeometryEncoder(
            self.config.patch_feature_dim,
            self.config.patch_hidden_dim,
            self.config.d_model,
            self.config.dropout,
        )
        self.position_encoding = build_position_encoding(
            self.config.position_encoding,
            grid_size=self.config.patch_grid_size,
            d_model=self.config.d_model,
            use_center_embedding=self.config.use_center_embedding,
        )
        self.blocks = nn.ModuleList([SpatialTransformerBlock(self.config) for _ in range(self.config.num_layers)])
        self.final_norm = nn.LayerNorm(self.config.d_model) if self.config.final_norm else nn.Identity()
        self.patch_decoder = nn.Linear(self.config.d_model, self.config.patch_size**2)
        self.port_decoder = nn.Linear(self.config.d_model, 4 * self.config.patch_size) if self.config.reconstruct_ports else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.patch_decoder.weight)
        nn.init.zeros_(self.patch_decoder.bias)
        if self.port_decoder is not None:
            nn.init.xavier_uniform_(self.port_decoder.weight)
            nn.init.zeros_(self.port_decoder.bias)

    def _build_geometry(self, halo_maps: torch.Tensor) -> PatchGeometry:
        return build_patch_geometry(
            halo_maps,
            core_size=self.config.core_size,
            patch_size=self.config.patch_size,
            num_cell_states=self.config.num_cell_states,
            free_state=self.config.free_state,
            unknown_state=self.config.unknown_state,
            include_port_openings=self.config.include_port_openings,
            include_outer_edge_mask=self.config.include_outer_edge_mask,
            include_port_known_mask=self.config.include_port_known_mask,
        )

    def forward(self, halo_maps: torch.Tensor, *, return_attention: bool = False) -> MapEncoderOutput:
        if halo_maps.ndim == 2:
            halo_maps = halo_maps.unsqueeze(0)
        geometry = self._build_geometry(halo_maps)
        parameter = next(self.parameters())
        features = geometry.features.to(device=parameter.device, dtype=parameter.dtype)
        connectivity_codes = geometry.connectivity_codes.to(device=parameter.device)

        tokens = self.patch_encoder(features)
        tokens = tokens + self.position_encoding(device=tokens.device, dtype=tokens.dtype)

        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            tokens, attention = block(tokens, connectivity_codes, return_attention=return_attention)
            if attention is not None:
                attention_maps.append(attention)
        tokens = self.final_norm(tokens)

        patch_logits_flat = self.patch_decoder(tokens)
        patch_logits = patch_logits_flat.reshape(
            halo_maps.shape[0],
            self.config.num_patch_tokens,
            self.config.patch_size,
            self.config.patch_size,
        )
        reconstruction_logits = unpatchify(
            patch_logits,
            grid_size=self.config.patch_grid_size,
            patch_size=self.config.patch_size,
        )
        port_logits = self.port_decoder(tokens) if self.port_decoder is not None else None
        return MapEncoderOutput(
            latent_tokens=tokens,
            reconstruction_logits=reconstruction_logits,
            patch_logits=patch_logits_flat,
            port_logits=port_logits,
            patch_geometry=geometry,
            attention_maps=tuple(attention_maps) if return_attention else None,
        )

    @torch.no_grad()
    def encode(self, halo_maps: torch.Tensor) -> torch.Tensor:
        return self.forward(halo_maps, return_attention=False).latent_tokens

    @property
    def num_latent_tokens(self) -> int:
        return self.config.num_patch_tokens
