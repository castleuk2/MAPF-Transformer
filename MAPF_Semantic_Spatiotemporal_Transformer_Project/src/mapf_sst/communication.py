from __future__ import annotations

import torch
from torch import nn

from .model import SemanticSpatiotemporalPolicy
from .types import CommunicationGraph, CommunicationOutput, PolicyBatch


class MultiRoundCommunicationPolicy(nn.Module):
    """Optional real message exchange among separately ego-centered views.

    Merely batching ego views does not create communication. This wrapper first
    generates one late-fusion self message per view, gathers the six selected
    neighbor messages with ``CommunicationGraph``, and performs one or more
    message-conditioned passes. The same action loss can backpropagate through
    all rounds.
    """

    def __init__(self, local_policy: SemanticSpatiotemporalPolicy) -> None:
        super().__init__()
        self.local_policy = local_policy

    @staticmethod
    def _gather_messages(
        messages: torch.Tensor, graph: CommunicationGraph
    ) -> torch.Tensor:
        index = graph.neighbor_view_index.clamp(0, messages.shape[0] - 1)
        gathered = messages[index]
        return gathered.masked_fill((~graph.neighbor_valid)[..., None], 0.0)

    def forward(
        self,
        batch: PolicyBatch,
        graph: CommunicationGraph,
        *,
        rounds: int = 1,
        return_tokens: bool = False,
    ) -> CommunicationOutput:
        if rounds < 1:
            raise ValueError("communication rounds must be at least one")
        first = self.local_policy(
            batch,
            coordination_mode="query_only",
            return_tokens=return_tokens,
        )
        if first.self_message is None:
            raise RuntimeError("query-only pass did not produce a self message")
        messages = first.self_message
        final = first
        for _ in range(rounds):
            neighbor_messages = self._gather_messages(messages, graph)
            final = self.local_policy(
                batch,
                coordination_mode="messages",
                neighbor_messages=neighbor_messages,
                neighbor_message_valid=graph.neighbor_valid,
                neighbor_message_source_slot=graph.neighbor_current_slot,
                return_tokens=return_tokens,
            )
            if final.self_message is None:
                raise RuntimeError("message-conditioned pass did not produce a self message")
            messages = final.self_message
        return CommunicationOutput(first_pass=first, final=final, rounds=rounds)
