from __future__ import annotations

import math

import torch
from torch import nn


class Learned2DPositionEncoding(nn.Module):
    def __init__(self, grid_size: int, d_model: int, use_center_embedding: bool = True) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        self.row_embedding = nn.Embedding(grid_size, d_model)
        self.col_embedding = nn.Embedding(grid_size, d_model)
        self.center_embedding = nn.Parameter(torch.zeros(d_model)) if use_center_embedding else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.row_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embedding.weight, std=0.02)
        if self.center_embedding is not None:
            nn.init.trunc_normal_(self.center_embedding, std=0.02)

    def forward(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        device = device or self.row_embedding.weight.device
        rows = torch.arange(self.grid_size, device=device)
        cols = torch.arange(self.grid_size, device=device)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        encoding = self.row_embedding(row_grid) + self.col_embedding(col_grid)
        if self.center_embedding is not None:
            center = self.grid_size // 2
            encoding = encoding.clone()
            encoding[center, center] = encoding[center, center] + self.center_embedding
        encoding = encoding.reshape(1, self.grid_size * self.grid_size, self.d_model)
        return encoding.to(dtype=dtype) if dtype is not None else encoding


class SineCosine2DPositionEncoding(nn.Module):
    def __init__(self, grid_size: int, d_model: int, use_center_embedding: bool = True) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sine/cosine encoding.")
        self.grid_size = grid_size
        self.d_model = d_model
        self.center_embedding = nn.Parameter(torch.zeros(d_model)) if use_center_embedding else None
        encoding = self._build_encoding(grid_size, d_model)
        self.register_buffer("encoding", encoding, persistent=False)
        if self.center_embedding is not None:
            nn.init.trunc_normal_(self.center_embedding, std=0.02)

    @staticmethod
    def _axis_encoding(positions: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = positions[:, None].to(torch.float32) * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    @classmethod
    def _build_encoding(cls, grid_size: int, d_model: int) -> torch.Tensor:
        axis_dim = d_model // 2
        coords = torch.arange(grid_size, dtype=torch.float32) - (grid_size - 1) / 2.0
        row = cls._axis_encoding(coords, axis_dim)
        col = cls._axis_encoding(coords, axis_dim)
        row_grid = row[:, None, :].expand(grid_size, grid_size, axis_dim)
        col_grid = col[None, :, :].expand(grid_size, grid_size, axis_dim)
        return torch.cat([row_grid, col_grid], dim=-1).reshape(1, grid_size * grid_size, d_model)

    def forward(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        encoding = self.encoding
        if device is not None:
            encoding = encoding.to(device=device)
        if self.center_embedding is not None:
            encoding = encoding.clone()
            center_index = (self.grid_size // 2) * self.grid_size + self.grid_size // 2
            encoding[:, center_index] = encoding[:, center_index] + self.center_embedding
        return encoding.to(dtype=dtype) if dtype is not None else encoding


def build_position_encoding(
    kind: str,
    *,
    grid_size: int,
    d_model: int,
    use_center_embedding: bool,
) -> nn.Module:
    if kind == "learned":
        return Learned2DPositionEncoding(grid_size, d_model, use_center_embedding)
    if kind == "sincos":
        return SineCosine2DPositionEncoding(grid_size, d_model, use_center_embedding)
    raise ValueError(f"Unsupported position encoding: {kind}")
