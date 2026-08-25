from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .constants import ACTION_DELTAS, ACTION_NAMES


@dataclass(slots=True)
class ResolutionResult:
    actions: torch.Tensor
    targets: torch.Tensor
    changed_from_argmax: torch.Tensor
    priorities: torch.Tensor


class CSPIBTResolver:
    """Single-step collision-safe priority-inheritance/backtracking resolver.

    It implements the deterministic resolver role used by the architecture, but
    is not claimed to be bit-exact with an external official CS-PIBT codebase.
    """

    def __init__(self, action_deltas: Iterable[tuple[int, int]] = ACTION_DELTAS) -> None:
        self.action_deltas = tuple(action_deltas)
        if len(self.action_deltas) != 5:
            raise ValueError("resolver expects WAIT, UP, DOWN, LEFT, RIGHT")

    @staticmethod
    def _inside(target: tuple[int, int], height: int, width: int) -> bool:
        return 0 <= target[0] < height and 0 <= target[1] < width

    def resolve(
        self,
        positions: torch.Tensor,
        preference_logits: torch.Tensor,
        obstacles: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        priorities: torch.Tensor | None = None,
        wait_age: torch.Tensor | None = None,
        on_goal: torch.Tensor | None = None,
    ) -> ResolutionResult:
        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape [N,2]")
        if preference_logits.shape != (positions.shape[0], 5):
            raise ValueError("preference_logits must have shape [N,5]")
        if obstacles.ndim != 2:
            raise ValueError("obstacles must have shape [H,W]")
        device = preference_logits.device
        n = positions.shape[0]
        valid = torch.ones(n, dtype=torch.bool, device=device) if valid is None else valid.bool()
        if priorities is None:
            wait_age = torch.zeros(n, device=device) if wait_age is None else wait_age.float()
            on_goal = torch.zeros(n, dtype=torch.bool, device=device) if on_goal is None else on_goal.bool()
            priorities = wait_age + (~on_goal).float() - torch.arange(n, device=device).float() * 1.0e-4
        priorities = priorities.float()

        pos_list = [tuple(map(int, positions[i].tolist())) for i in range(n)]
        pos_to_agent = {pos_list[i]: i for i in range(n) if bool(valid[i])}
        if len(pos_to_agent) != int(valid.sum()):
            raise ValueError("valid agents must occupy unique cells")
        height, width = map(int, obstacles.shape)
        action_orders = torch.argsort(preference_logits, dim=-1, descending=True).tolist()
        assignments: dict[int, int] = {}
        targets: dict[int, tuple[int, int]] = {}
        reserved: dict[tuple[int, int], int] = {}

        def edge_swap(agent: int, target: tuple[int, int]) -> bool:
            current = pos_list[agent]
            return any(
                other != agent and other_target == current and target == pos_list[other]
                for other, other_target in targets.items()
            )

        def plan(agent: int) -> bool:
            if agent in assignments:
                return True
            current = pos_list[agent]
            for action in action_orders[agent]:
                dr, dc = self.action_deltas[action]
                target = (current[0] + dr, current[1] + dc)
                if not self._inside(target, height, width) or int(obstacles[target].item()) != 0:
                    continue
                if target in reserved and reserved[target] != agent:
                    continue
                snapshot = (assignments.copy(), targets.copy(), reserved.copy())
                assignments[agent] = int(action); targets[agent] = target; reserved[target] = agent
                occupant = pos_to_agent.get(target)
                feasible = True
                if occupant is not None and occupant != agent:
                    if occupant not in assignments:
                        feasible = plan(occupant)
                    if feasible and targets.get(occupant, pos_list[occupant]) == target:
                        feasible = False
                if feasible and edge_swap(agent, target):
                    feasible = False
                if feasible:
                    return True
                assignments.clear(); assignments.update(snapshot[0])
                targets.clear(); targets.update(snapshot[1])
                reserved.clear(); reserved.update(snapshot[2])
            return False

        order = torch.argsort(priorities.masked_fill(~valid, -1.0e9), descending=True).tolist()
        for agent in order:
            if bool(valid[agent]) and agent not in assignments and not plan(agent):
                raise RuntimeError(f"failed to find a collision-free action for agent {agent}")
        actions = torch.zeros(n, dtype=torch.long, device=device)
        target_tensor = positions.clone().long()
        for i in range(n):
            if i in assignments:
                actions[i] = assignments[i]
                target_tensor[i] = torch.tensor(targets[i], device=device)
        changed = valid & (actions != preference_logits.argmax(dim=-1))
        self.assert_collision_free(positions, actions, obstacles, valid=valid)
        return ResolutionResult(actions, target_tensor, changed, priorities)

    def assert_collision_free(self, positions: torch.Tensor, actions: torch.Tensor, obstacles: torch.Tensor, *, valid: torch.Tensor | None = None) -> None:
        valid = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device) if valid is None else valid.bool()
        current: list[tuple[int, int]] = []
        targets: list[tuple[int, int]] = []
        h, w = map(int, obstacles.shape)
        for i in range(positions.shape[0]):
            if not bool(valid[i]):
                continue
            pos = tuple(map(int, positions[i].tolist()))
            dr, dc = self.action_deltas[int(actions[i])]
            target = (pos[0] + dr, pos[1] + dc)
            if not self._inside(target, h, w) or int(obstacles[target].item()) != 0:
                raise AssertionError(f"invalid target for agent {i}")
            current.append(pos); targets.append(target)
        if len(set(targets)) != len(targets):
            raise AssertionError("vertex conflict detected")
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                if targets[i] == current[j] and targets[j] == current[i]:
                    raise AssertionError("edge-swap conflict detected")

    @staticmethod
    def format_actions(actions: torch.Tensor) -> list[str]:
        return [ACTION_NAMES[int(action)] for action in actions.tolist()]
