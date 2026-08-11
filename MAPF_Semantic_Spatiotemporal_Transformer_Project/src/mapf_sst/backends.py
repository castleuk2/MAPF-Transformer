from __future__ import annotations

from typing import Protocol

import torch


class JointActionBackend(Protocol):
    """Optional consumer of per-agent five-action preferences.

    Tokenization and preference learning are intentionally independent of any
    specific MAPF resolver. A backend may implement independent argmax, learned
    negotiation, PIBT-family coordination, LaCAM configuration generation, or
    another application-specific joint-action rule.
    """

    def resolve(self, preferences: torch.Tensor, **context: torch.Tensor) -> torch.Tensor:
        """Return actions [N] from preferences [N,5]."""


class IndependentArgmaxBackend:
    """Reference backend only; it does not guarantee collision-free actions."""

    def resolve(self, preferences: torch.Tensor, **context: torch.Tensor) -> torch.Tensor:
        if preferences.ndim != 2 or preferences.shape[-1] != 5:
            raise ValueError("preferences must have shape [N,5]")
        return preferences.argmax(dim=-1)
