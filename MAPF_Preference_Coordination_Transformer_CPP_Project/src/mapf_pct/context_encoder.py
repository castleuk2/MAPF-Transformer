from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .types import PolicyBatch


class EventEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        f = max(8, d // 8)
        self.action_embedding = nn.Embedding(config.num_actions + 1, f)
        self.outcome_embedding = nn.Embedding(config.outcome_states, f)
        self.numeric_projection = nn.Sequential(
            nn.Linear(config.event_numeric_dim, f * 2),
            nn.GELU(),
            nn.Linear(f * 2, f),
        )
        self.projection = nn.Sequential(
            nn.Linear(4 * f, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.lag_embedding = nn.Embedding(config.event_tokens, d)
        self.event_type = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.event_type, std=0.02)

    def forward(self, batch: PolicyBatch) -> torch.Tensor:
        action = batch.event_action.clamp(0, self.config.num_actions)
        executed = batch.event_executed.clamp(0, self.config.num_actions)
        outcome = batch.event_outcome.clamp(0, self.config.outcome_states - 1)
        raw = torch.cat(
            (
                self.action_embedding(action),
                self.action_embedding(executed),
                self.outcome_embedding(outcome),
                self.numeric_projection(batch.event_numeric.float()),
            ),
            dim=-1,
        )
        tokens = self.projection(raw)
        lag = self.lag_embedding(torch.arange(self.config.event_tokens, device=tokens.device))
        tokens = tokens + lag[None] + self.event_type
        return tokens.masked_fill((~batch.event_valid)[..., None], 0.0)


class SceneTokenEncoder(nn.Module):
    """Creates the single global scene token.

    The initial token carries global, inference-available summaries and problem
    semantics. During bidirectional coordination attention it becomes a global
    read/write workspace: it aggregates map/agent/relation/event information and
    broadcasts the resulting scene regime back to local decision tokens.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.base_token = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.base_token, std=0.02)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(config.scene_numeric_dim, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
        )
        self.task_mode_embedding = nn.Embedding(2, d)
        self.target_mode_embedding = nn.Embedding(2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch: PolicyBatch) -> torch.Tensor:
        token = (
            self.base_token[None]
            + self.numeric_mlp(batch.scene_numeric.float())
            + self.task_mode_embedding(batch.task_mode.long().clamp(0, 1))
            + self.target_mode_embedding(batch.target_mode.long().clamp(0, 1))
        )
        return self.norm(token).unsqueeze(1)
