from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterator

import torch


@dataclass(slots=True)
class PolicyBatch:
    """All tensors required to construct the fixed 256-token policy context.

    A single unbatched sample is accepted by ``stack_policy_batches``; model input
    should always include a leading batch dimension.
    """

    local_maps: torch.Tensor
    agent_xy: torch.Tensor
    goal_delta: torch.Tensor
    remaining_hops: torch.Tensor
    agent_valid: torch.Tensor
    track_reset: torch.Tensor
    on_goal: torch.Tensor
    goal_outside: torch.Tensor

    history_selected: torch.Tensor
    history_executed: torch.Tensor
    history_outcome: torch.Tensor
    history_delta_ctg: torch.Tensor
    history_valid: torch.Tensor

    candidate_target_xy: torch.Tensor
    candidate_in_view: torch.Tensor
    candidate_delta_ctg: torch.Tensor
    candidate_greedy: torch.Tensor
    candidate_static_free: torch.Tensor
    candidate_target_occupied: torch.Tensor
    candidate_contenders: torch.Tensor
    candidate_edge_swap: torch.Tensor
    candidate_bottleneck: torch.Tensor
    candidate_congestion: torch.Tensor

    event_action: torch.Tensor
    event_executed: torch.Tensor
    event_outcome: torch.Tensor
    event_numeric: torch.Tensor
    event_valid: torch.Tensor

    scene_numeric: torch.Tensor
    task_mode: torch.Tensor
    target_mode: torch.Tensor

    all_agent_actions: torch.Tensor | None = None
    action_soft_targets: torch.Tensor | None = None
    reason_labels: torch.Tensor | None = None
    reason_valid: torch.Tensor | None = None
    scene_risk_labels: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "PolicyBatch":
        values = {
            field.name: (
                getattr(self, field.name).to(device)
                if isinstance(getattr(self, field.name), torch.Tensor)
                else getattr(self, field.name)
            )
            for field in fields(self)
        }
        return replace(self, **values)

    @property
    def batch_size(self) -> int:
        return int(self.local_maps.shape[0])

    def tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                yield field.name, value


def stack_policy_batches(samples: list[PolicyBatch]) -> PolicyBatch:
    if not samples:
        raise ValueError("cannot stack an empty sample list")
    values = {}
    for field in fields(PolicyBatch):
        items = [getattr(sample, field.name) for sample in samples]
        if items[0] is None:
            if any(item is not None for item in items):
                raise ValueError(f"mixed None/non-None values for {field.name}")
            values[field.name] = None
        else:
            values[field.name] = torch.stack(items, dim=0)
    return PolicyBatch(**values)
