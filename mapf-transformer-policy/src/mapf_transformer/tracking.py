from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class TrackingResult:
    agent_ids: np.ndarray
    valid: np.ndarray
    reset: np.ndarray


class StableNeighborTracker:
    """Maintains stable local neighbor slots without using future information."""

    def __init__(self, max_neighbors: int = 15, grace_steps: int = 1) -> None:
        self.max_neighbors = int(max_neighbors)
        self.grace_steps = int(grace_steps)
        self.slot_ids = np.full(self.max_neighbors, -1, dtype=np.int32)
        self.missing_steps = np.zeros(self.max_neighbors, dtype=np.int16)

    def reset(self) -> None:
        self.slot_ids.fill(-1)
        self.missing_steps.fill(0)

    def update(
        self,
        visible_agent_ids: Sequence[int],
        candidate_scores: Mapping[int, float] | None = None,
    ) -> TrackingResult:
        visible = {int(agent_id) for agent_id in visible_agent_ids}
        scores = candidate_scores or {}
        reset_flags = np.zeros(self.max_neighbors, dtype=bool)

        # Preserve visible tracks and reserve briefly missing tracks.
        for slot, agent_id in enumerate(self.slot_ids.tolist()):
            if agent_id < 0:
                continue
            if agent_id in visible:
                self.missing_steps[slot] = 0
            else:
                self.missing_steps[slot] += 1
                if self.missing_steps[slot] > self.grace_steps:
                    self.slot_ids[slot] = -1
                    self.missing_steps[slot] = 0

        assigned = {int(agent_id) for agent_id in self.slot_ids.tolist() if agent_id >= 0}
        unassigned_visible = sorted(
            visible - assigned,
            key=lambda agent_id: (float(scores.get(agent_id, np.inf)), agent_id),
        )

        for agent_id in unassigned_visible:
            free_slots = np.flatnonzero(self.slot_ids < 0)
            if free_slots.size == 0:
                break
            slot = int(free_slots[0])
            self.slot_ids[slot] = agent_id
            self.missing_steps[slot] = 0
            reset_flags[slot] = True

        valid = np.asarray([agent_id in visible for agent_id in self.slot_ids.tolist()], dtype=bool)
        return TrackingResult(self.slot_ids.copy(), valid, reset_flags)
