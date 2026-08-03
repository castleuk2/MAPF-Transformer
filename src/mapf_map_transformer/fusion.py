from __future__ import annotations

import math

import torch
from torch import nn


class AgentMapCrossAttention(nn.Module):
    """Optional adapter: agent tokens query the 25 structured map tokens."""

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.norm_agent = nn.LayerNorm(d_model)
        self.norm_map = nn.LayerNorm(d_model)
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 27, 27))
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        patch_centers = torch.tensor([-6, -3, 0, 3, 6], dtype=torch.long)
        rows, cols = torch.meshgrid(patch_centers, patch_centers, indexing="ij")
        # x grows rightward; y grows upward. Array rows therefore use -rows.
        self.register_buffer("patch_xy", torch.stack([cols.reshape(-1), -rows.reshape(-1)], dim=-1), persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        agent_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, num_agents, _ = agent_tokens.shape
        if map_tokens.shape[:2] != (batch, 25):
            raise ValueError("map_tokens must have shape [B,25,D].")
        query = self.q_projection(self.norm_agent(agent_tokens))
        key = self.k_projection(self.norm_map(map_tokens))
        value = self.v_projection(self.norm_map(map_tokens))
        query = query.reshape(batch, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, 25, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, 25, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if agent_xy is not None:
            if agent_xy.shape != (batch, num_agents, 2):
                raise ValueError("agent_xy must have shape [B,A,2].")
            relative = self.patch_xy.to(agent_xy.device)[None, None, :, :] - agent_xy[:, :, None, :].to(torch.long)
            relative = relative.clamp(-13, 13) + 13
            dx = relative[..., 0]
            dy = relative[..., 1]
            bias = self.relative_bias[:, dy, dx].permute(1, 0, 2, 3)
            logits = logits + bias.to(dtype=logits.dtype)

        attention = torch.softmax(logits, dim=-1)
        attention = self.attention_dropout(attention)
        update = torch.matmul(attention, value).transpose(1, 2).reshape(batch, num_agents, self.d_model)
        agent_tokens = agent_tokens + self.output_dropout(self.output_projection(update))
        agent_tokens = agent_tokens + self.ffn(self.norm_ffn(agent_tokens))
        return agent_tokens
