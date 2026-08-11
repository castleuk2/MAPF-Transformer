from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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


class MultiheadAttentionWithBias(nn.Module):
    """Batch-first multi-head attention with additive score bias.

    ``attn_bias`` may have shape [H,Q,K] or [B,H,Q,K]. Positive values are
    soft priors; large negative values can be used as deterministic routing masks.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        query_padding_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        # Keep the explicit path when weights are requested for diagnostics.
        # Training uses PyTorch's fused scaled-dot-product implementation.
        if not return_weights:
            additive_mask = None
            if attn_bias is not None:
                if attn_bias.ndim == 3:
                    attn_bias = attn_bias.unsqueeze(0)
                additive_mask = attn_bias.to(dtype=q.dtype, device=q.device)
            if key_padding_mask is not None:
                padding = key_padding_mask.bool()[:, None, None, :]
                if additive_mask is None:
                    additive_mask = torch.zeros(
                        q.shape[0], 1, q.shape[-2], k.shape[-2],
                        dtype=q.dtype, device=q.device,
                    )
                additive_mask = additive_mask.masked_fill(
                    padding, torch.finfo(q.dtype).min
                )
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=additive_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).contiguous().view(
                query.shape[0], query.shape[1], self.d_model
            )
            output = self.out_proj(attended)
            if query_padding_mask is not None:
                output = output.masked_fill(query_padding_mask.bool()[..., None], 0.0)
            return output, None

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            if attn_bias.ndim == 3:
                attn_bias = attn_bias.unsqueeze(0)
            scores = scores + attn_bias.to(dtype=scores.dtype, device=scores.device)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.bool()[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        attended = torch.matmul(self.dropout(weights), v)
        attended = attended.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.d_model
        )
        output = self.out_proj(attended)
        if query_padding_mask is not None:
            output = output.masked_fill(query_padding_mask.bool()[..., None], 0.0)
        return output, weights if return_weights else None


class PreNormAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttentionWithBias(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            query_padding_mask=key_padding_mask,
        )
        x = x + attended
        x = x + self.ff(self.norm2(x))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask[..., None], 0.0)
        return x
