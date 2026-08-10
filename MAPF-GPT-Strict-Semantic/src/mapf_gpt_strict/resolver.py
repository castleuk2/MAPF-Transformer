from __future__ import annotations

import numpy as np
import torch


ACTION_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class CSPIBTResolver:
    """Deterministic single-step priority-inheritance/backtracking resolver."""

    def resolve(self, positions, preference_logits, obstacles, *, wait_age=None, on_goal=None):
        positions = torch.as_tensor(positions, dtype=torch.long)
        logits = torch.as_tensor(preference_logits, dtype=torch.float32)
        obstacles = torch.as_tensor(obstacles, dtype=torch.long)
        n = len(positions)
        wait_age = torch.zeros(n) if wait_age is None else torch.as_tensor(wait_age).float()
        on_goal = torch.zeros(n, dtype=torch.bool) if on_goal is None else torch.as_tensor(on_goal).bool()
        priorities = wait_age + (~on_goal).float() - torch.arange(n).float() * 1e-4
        order = torch.argsort(priorities, descending=True).tolist()
        choices = torch.argsort(logits, dim=-1, descending=True).tolist()
        pos = [tuple(map(int, item)) for item in positions.tolist()]
        occupant = {cell: agent for agent, cell in enumerate(pos)}
        assigned: dict[int, int] = {}
        targets: dict[int, tuple[int, int]] = {}
        reserved: dict[tuple[int, int], int] = {}
        height, width = obstacles.shape

        def plan(agent: int, active: set[int]) -> bool:
            if agent in assigned:
                return True
            if agent in active:
                return False
            active = active | {agent}
            row, col = pos[agent]
            for action in choices[agent]:
                dr, dc = ACTION_DELTAS[action]; target = (row + dr, col + dc)
                if not (0 <= target[0] < height and 0 <= target[1] < width):
                    continue
                if int(obstacles[target]) != 0 or target in reserved:
                    continue
                snapshot = (assigned.copy(), targets.copy(), reserved.copy())
                assigned[agent] = action; targets[agent] = target; reserved[target] = agent
                other = occupant.get(target)
                feasible = other is None or other == agent or plan(other, active)
                if feasible and other is not None and other != agent and targets.get(other, pos[other]) == target:
                    feasible = False
                if feasible and any(
                    index != agent and other_target == pos[agent] and target == pos[index]
                    for index, other_target in targets.items()
                ):
                    feasible = False
                if feasible:
                    return True
                assigned.clear(); assigned.update(snapshot[0])
                targets.clear(); targets.update(snapshot[1])
                reserved.clear(); reserved.update(snapshot[2])
            return False

        for agent in order:
            if agent not in assigned and not plan(agent, set()):
                raise RuntimeError(f"no collision-free action for agent {agent}")
        return np.asarray([assigned.get(agent, 0) for agent in range(n)], dtype=np.int64)
