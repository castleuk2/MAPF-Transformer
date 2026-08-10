from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from ..config import ModelConfig
from ..constants import Action, DeltaCTG, Outcome
from ..types import PolicyBatch


def _bottleneck(core: torch.Tensor, row: int, col: int) -> bool:
    if not (0 <= row < core.shape[0] and 0 <= col < core.shape[1]):
        return False
    if int(core[row, col]) != 0:
        return False
    degree = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < core.shape[0] and 0 <= cc < core.shape[1] and int(core[rr, cc]) == 0:
            degree += 1
    return degree <= 2


def make_synthetic_sample(config: ModelConfig, seed: int) -> PolicyBatch:
    """Generate a shape-correct, weakly learnable smoke-test sample.

    This generator is deliberately not an MAPF expert. Real experiments should
    use the NPZ adapter with MAPF-LNS2/LaCAM/other expert trajectories.
    """
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    n, h, a = config.max_agents, config.history_steps, config.num_actions
    core_size = config.core_map_size

    local_map = (torch.rand(config.local_map_size, config.local_map_size, generator=generator) < 0.16).long()
    # Keep the halo mostly blocked and guarantee enough free core cells.
    local_map[0] = 1
    local_map[-1] = 1
    local_map[:, 0] = 1
    local_map[:, -1] = 1
    core = local_map[1:-1, 1:-1]
    free = torch.nonzero(core == 0, as_tuple=False)
    if free.shape[0] < n:
        core.zero_()
        free = torch.nonzero(core == 0, as_tuple=False)
    permutation = torch.randperm(free.shape[0], generator=generator)
    valid_count = rng.randint(4, n)
    chosen = free[permutation[:valid_count]]

    pad_coord = config.core_map_size + 1
    agent_xy = torch.full((n, 2), pad_coord, dtype=torch.long)
    agent_xy[:valid_count] = chosen
    agent_valid = torch.zeros(n, dtype=torch.bool)
    agent_valid[:valid_count] = True
    track_reset = torch.zeros(n, dtype=torch.bool)
    if valid_count > 1:
        track_reset[1:valid_count] = torch.rand(valid_count - 1, generator=generator) < 0.05

    goal_delta = torch.zeros(n, 2, dtype=torch.long)
    goal_delta[:valid_count] = torch.randint(
        -config.goal_delta_clip,
        config.goal_delta_clip + 1,
        (valid_count, 2),
        generator=generator,
    )
    on_goal = torch.zeros(n, dtype=torch.bool)
    on_goal[:valid_count] = torch.rand(valid_count, generator=generator) < 0.08
    goal_delta[on_goal] = 0
    goal_outside = goal_delta.abs().amax(dim=-1) > (core_size // 2)
    remaining_hops = torch.full((n,), config.max_hops + 2, dtype=torch.long)
    remaining_hops[:valid_count] = goal_delta[:valid_count].abs().sum(dim=-1).clamp_max(config.max_hops)

    history_selected = torch.full((n, h), int(Action.PAD), dtype=torch.long)
    history_executed = torch.full((n, h), int(Action.PAD), dtype=torch.long)
    history_outcome = torch.full((n, h), int(Outcome.PAD), dtype=torch.long)
    history_delta = torch.full((n, h), int(DeltaCTG.PAD), dtype=torch.long)
    history_valid = torch.zeros(n, h, dtype=torch.bool)
    history_length = rng.randint(1, h)
    for i in range(valid_count):
        history_valid[i, h - history_length :] = True
        selected = torch.randint(0, a, (history_length,), generator=generator)
        executed = selected.clone()
        failed = torch.rand(history_length, generator=generator) < 0.12
        executed[failed] = int(Action.WAIT)
        history_selected[i, h - history_length :] = selected
        history_executed[i, h - history_length :] = executed
        history_outcome[i, h - history_length :] = torch.where(
            failed,
            torch.full_like(selected, int(Outcome.BLOCKED_OR_OVERRIDDEN)),
            torch.where(
                selected == int(Action.WAIT),
                torch.full_like(selected, int(Outcome.WAIT)),
                torch.full_like(selected, int(Outcome.SUCCESS)),
            ),
        )
        history_delta[i, h - history_length :] = torch.randint(0, 3, (history_length,), generator=generator)

    deltas = torch.tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)), dtype=torch.long)
    raw_targets = agent_xy[:, None, :] + deltas[None]
    candidate_in_view = (
        (raw_targets[..., 0] >= 0)
        & (raw_targets[..., 0] < core_size)
        & (raw_targets[..., 1] >= 0)
        & (raw_targets[..., 1] < core_size)
        & agent_valid[:, None]
    )
    candidate_target_xy = raw_targets.clamp(0, core_size - 1)
    candidate_static_free = torch.zeros(n, a, dtype=torch.bool)
    for i in range(valid_count):
        for action in range(a):
            if candidate_in_view[i, action]:
                rr, cc = map(int, candidate_target_xy[i, action].tolist())
                candidate_static_free[i, action] = int(core[rr, cc]) == 0

    current_dist = goal_delta.abs().sum(dim=-1)
    target_goal_delta = goal_delta[:, None, :] - deltas[None]
    target_dist = target_goal_delta.abs().sum(dim=-1)
    candidate_delta_ctg = torch.full((n, a), int(DeltaCTG.PAD), dtype=torch.long)
    candidate_delta_ctg[agent_valid[:, None] & ~candidate_static_free] = int(DeltaCTG.BLOCKED)
    feasible = agent_valid[:, None] & candidate_static_free
    candidate_delta_ctg[feasible & (target_dist < current_dist[:, None])] = int(DeltaCTG.DECREASE)
    candidate_delta_ctg[feasible & (target_dist == current_dist[:, None])] = int(DeltaCTG.SAME)
    candidate_delta_ctg[feasible & (target_dist > current_dist[:, None])] = int(DeltaCTG.INCREASE)
    candidate_greedy = candidate_delta_ctg == int(DeltaCTG.DECREASE)
    candidate_greedy[on_goal, int(Action.WAIT)] = True

    occupied = {tuple(agent_xy[i].tolist()): i for i in range(valid_count)}
    candidate_target_occupied = torch.zeros(n, a, dtype=torch.bool)
    candidate_bottleneck = torch.zeros(n, a, dtype=torch.bool)
    candidate_congestion = torch.zeros(n, a, dtype=torch.float32)
    for i in range(valid_count):
        for action in range(a):
            target = tuple(candidate_target_xy[i, action].tolist())
            candidate_target_occupied[i, action] = target in occupied and occupied[target] != i
            candidate_bottleneck[i, action] = _bottleneck(core, target[0], target[1])
            nearby = 0
            for j in range(valid_count):
                if j != i and sum(abs(int(agent_xy[j, k]) - target[k]) for k in (0, 1)) <= 2:
                    nearby += 1
            candidate_congestion[i, action] = nearby / max(1, valid_count - 1)

    candidate_contenders = torch.zeros(n, a, dtype=torch.long)
    for i in range(valid_count):
        for action in range(a):
            target = candidate_target_xy[i, action]
            count = 0
            for j in range(valid_count):
                if j == i:
                    continue
                for other_action in range(a):
                    if candidate_greedy[j, other_action] and torch.equal(
                        candidate_target_xy[j, other_action], target
                    ):
                        count += 1
            candidate_contenders[i, action] = min(count, config.contender_buckets - 1)

    candidate_edge_swap = torch.zeros(n, a, dtype=torch.bool)
    for i in range(valid_count):
        for action in range(a):
            target = candidate_target_xy[i, action]
            for j in range(valid_count):
                if i == j or not torch.equal(target, agent_xy[j]):
                    continue
                for other_action in range(a):
                    if candidate_greedy[j, other_action] and torch.equal(
                        candidate_target_xy[j, other_action], agent_xy[i]
                    ):
                        candidate_edge_swap[i, action] = True

    # Weak synthetic expert: prefer feasible progress, then low conflict/congestion, then WAIT.
    labels = torch.zeros(n, dtype=torch.long)
    reason_labels = torch.zeros(n, config.reason_classes, dtype=torch.float32)
    reason_valid = agent_valid.clone()
    for i in range(valid_count):
        score = torch.full((a,), -1.0e4)
        for action in range(a):
            if candidate_static_free[i, action]:
                progress = {
                    int(DeltaCTG.DECREASE): 3.0,
                    int(DeltaCTG.SAME): 1.0,
                    int(DeltaCTG.INCREASE): 0.0,
                }.get(int(candidate_delta_ctg[i, action]), -10.0)
                score[action] = (
                    progress
                    - 1.5 * float(candidate_contenders[i, action])
                    - 2.0 * float(candidate_edge_swap[i, action])
                    - float(candidate_congestion[i, action])
                )
        if on_goal[i]:
            labels[i] = int(Action.WAIT)
            reason_labels[i, 1] = 1.0
        else:
            labels[i] = int(score.argmax())
            chosen = labels[i]
            if candidate_delta_ctg[i, chosen] == int(DeltaCTG.DECREASE):
                reason_labels[i, 0] = 1.0
            if chosen == int(Action.WAIT) and candidate_contenders[i].max() > 0:
                reason_labels[i, 5] = 1.0
            if candidate_bottleneck[i, chosen]:
                reason_labels[i, 6] = 1.0
            if candidate_congestion[i, chosen] < candidate_congestion[i].mean():
                reason_labels[i, 7] = 1.0
            recent_waits = int((history_selected[i] == int(Action.WAIT)).sum())
            if recent_waits >= 3 and chosen != int(Action.WAIT):
                reason_labels[i, 9] = 1.0

    event_action = torch.full((h,), int(Action.PAD), dtype=torch.long)
    event_executed = torch.full((h,), int(Action.PAD), dtype=torch.long)
    event_outcome = torch.full((h,), int(Outcome.PAD), dtype=torch.long)
    event_numeric = torch.zeros(h, config.event_numeric_dim, dtype=torch.float32)
    event_valid = history_valid[0].clone()
    event_action[event_valid] = history_selected[0, event_valid]
    event_executed[event_valid] = history_executed[0, event_valid]
    event_outcome[event_valid] = history_outcome[0, event_valid]
    for step in range(h):
        if not event_valid[step]:
            continue
        moved_ratio = float((history_executed[:, step] != int(Action.WAIT))[agent_valid].float().mean())
        wait_ratio = float((history_selected[:, step] == int(Action.WAIT))[agent_valid].float().mean())
        overridden = float((history_selected[:, step] != history_executed[:, step])[agent_valid].float().mean())
        event_numeric[step] = torch.tensor(
            (
                moved_ratio,
                wait_ratio,
                overridden,
                valid_count / n,
                float((history_delta[:, step] == int(DeltaCTG.DECREASE))[agent_valid].float().mean()),
                float(candidate_contenders[:valid_count].max()) / max(1, config.contender_buckets - 1),
                float(candidate_edge_swap[:valid_count].float().mean()),
                float(on_goal[:valid_count].float().mean()),
            )
        )

    valid_hops = remaining_hops[:valid_count].float() / max(1, config.max_hops)
    scene_numeric = torch.tensor(
        (
            float(core.float().mean()),
            valid_count / n,
            float(on_goal[:valid_count].float().mean()),
            float(valid_hops.mean()),
            float(valid_hops.std(unbiased=False)),
            float(valid_hops.min()),
            float(valid_hops.max()),
            float((history_selected[:valid_count, -1] == int(Action.WAIT)).float().mean()),
            float((history_selected[:valid_count, -1] != history_executed[:valid_count, -1]).float().mean()),
            float(candidate_static_free[:valid_count].float().mean()),
            float(candidate_greedy[:valid_count].float().mean()),
            float((candidate_contenders[:valid_count] > 0).float().mean()),
            float(candidate_edge_swap[:valid_count].float().mean()),
            float(candidate_bottleneck[:valid_count].float().mean()),
            float(candidate_congestion[:valid_count].mean()),
            float(history_valid[:valid_count].float().mean()),
        ),
        dtype=torch.float32,
    )
    scene_risk = torch.tensor(
        2
        if scene_numeric[11] + scene_numeric[12] > 0.35
        else 1 if scene_numeric[11] + scene_numeric[12] > 0.12 else 0,
        dtype=torch.long,
    )

    return PolicyBatch(
        local_maps=local_map,
        agent_xy=agent_xy,
        goal_delta=goal_delta,
        remaining_hops=remaining_hops,
        agent_valid=agent_valid,
        track_reset=track_reset,
        on_goal=on_goal,
        goal_outside=goal_outside,
        history_selected=history_selected,
        history_executed=history_executed,
        history_outcome=history_outcome,
        history_delta_ctg=history_delta,
        history_valid=history_valid,
        candidate_target_xy=candidate_target_xy,
        candidate_in_view=candidate_in_view,
        candidate_delta_ctg=candidate_delta_ctg,
        candidate_greedy=candidate_greedy,
        candidate_static_free=candidate_static_free,
        candidate_target_occupied=candidate_target_occupied,
        candidate_contenders=candidate_contenders,
        candidate_edge_swap=candidate_edge_swap,
        candidate_bottleneck=candidate_bottleneck,
        candidate_congestion=candidate_congestion,
        event_action=event_action,
        event_executed=event_executed,
        event_outcome=event_outcome,
        event_numeric=event_numeric,
        event_valid=event_valid,
        scene_numeric=scene_numeric,
        task_mode=torch.tensor(0, dtype=torch.long),
        target_mode=torch.tensor(0, dtype=torch.long),
        all_agent_actions=labels,
        action_soft_targets=None,
        reason_labels=reason_labels,
        reason_valid=reason_valid,
        scene_risk_labels=scene_risk,
    )


class SyntheticPolicyDataset(Dataset[PolicyBatch]):
    def __init__(self, config: ModelConfig, size: int, seed: int = 0) -> None:
        self.config = config
        self.size = int(size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> PolicyBatch:
        return make_synthetic_sample(self.config, self.seed + int(index) * 104729)
