from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import Dataset

from ..config import ModelConfig
from ..constants import ACTION_DELTAS, Action, DeltaCTG
from ..types import CommunicationGraph, PolicyBatch


def _delta_category(delta: torch.Tensor, static_free: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(delta, int(DeltaCTG.SAME), dtype=torch.long)
    result = torch.where(delta < 0, torch.full_like(result, int(DeltaCTG.DECREASE)), result)
    result = torch.where(delta > 0, torch.full_like(result, int(DeltaCTG.INCREASE)), result)
    result = torch.where(
        ~static_free, torch.full_like(result, int(DeltaCTG.BLOCKED)), result
    )
    return result


def make_synthetic_policy_batch(
    config: ModelConfig,
    batch_size: int = 2,
    *,
    seed: int = 0,
    num_current_agents: int | None = None,
) -> PolicyBatch:
    generator = torch.Generator().manual_seed(seed)
    b = batch_size
    n = config.max_current_agents
    h = config.history_tracks
    t = config.history_steps
    a = config.num_actions
    center = config.core_map_size // 2

    local_maps = (torch.rand(b, 17, 17, generator=generator) < 0.16).long()
    local_maps[:, center + 1, center + 1] = 0

    count = num_current_agents if num_current_agents is not None else n
    count = max(1, min(n, count))
    current_valid = torch.zeros(b, n, dtype=torch.bool)
    current_valid[:, :count] = True
    current_track_reset = torch.zeros(b, n, dtype=torch.bool)
    current_track_reset[:, 1:count] = torch.rand(
        b, max(count - 1, 0), generator=generator
    ) < 0.05
    current_global_ids = torch.full((b, n), -1, dtype=torch.long)
    current_global_ids[:, :count] = torch.arange(count)

    current_xy = torch.zeros(b, n, 2, dtype=torch.long)
    if count > 1:
        current_xy[:, 1:count] = torch.randint(
            -center, center + 1, (b, count - 1, 2), generator=generator
        )
    current_goal_delta = torch.randint(
        -20, 21, (b, n, 2), generator=generator
    )
    current_goal_delta[:, 0] = torch.tensor([5, 5])
    current_goal_delta = current_goal_delta.masked_fill((~current_valid)[..., None], 0)
    detour = torch.randint(0, 5, (b, n), generator=generator)
    current_hops = current_goal_delta.abs().sum(dim=-1) + detour
    current_hops = current_hops.masked_fill(~current_valid, config.max_hops + 2)

    action_delta = torch.tensor(ACTION_DELTAS, dtype=torch.long)
    source_core = current_xy + center
    target_core = source_core[:, :, None, :] + action_delta[None, None]
    in_halo = (
        (target_core[..., 0] >= -1)
        & (target_core[..., 0] <= config.core_map_size)
        & (target_core[..., 1] >= -1)
        & (target_core[..., 1] <= config.core_map_size)
    )
    target_local_map = target_core + 1
    rr = target_local_map[..., 0].clamp(0, 16)
    cc = target_local_map[..., 1].clamp(0, 16)
    batch_index = torch.arange(b)[:, None, None]
    static_free = in_halo & (local_maps[batch_index, rr, cc] == 0)
    static_free &= current_valid[:, :, None]
    static_free[:, :, int(Action.WAIT)] = current_valid
    in_core = (
        (target_core[..., 0] >= 0)
        & (target_core[..., 0] < config.core_map_size)
        & (target_core[..., 1] >= 0)
        & (target_core[..., 1] < config.core_map_size)
    )

    new_goal_delta = current_goal_delta[:, :, None, :] - action_delta[None, None]
    one_hop = new_goal_delta.abs().sum(dim=-1) + detour[:, :, None]
    one_hop = one_hop.masked_fill(~static_free, config.max_hops + 1)
    raw_delta = one_hop - current_hops[:, :, None]
    delta_ctg = _delta_category(raw_delta, static_free)
    greedy = static_free & (one_hop < current_hops[:, :, None])
    on_goal = current_hops == 0
    greedy[:, :, int(Action.WAIT)] |= on_goal

    # Degree-based bottleneck flag in the local map.
    bottleneck = torch.zeros(b, n, a, dtype=torch.bool)
    for action in range(a):
        tr = target_local_map[:, :, action, 0]
        tc = target_local_map[:, :, action, 1]
        degree = torch.zeros(b, n, dtype=torch.long)
        for dr, dc in ACTION_DELTAS[1:]:
            nr = (tr + dr).clamp(0, 16)
            nc = (tc + dc).clamp(0, 16)
            inside = (tr + dr >= 0) & (tr + dr <= 16) & (tc + dc >= 0) & (tc + dc <= 16)
            degree += (inside & (local_maps[torch.arange(b)[:, None], nr, nc] == 0)).long()
        bottleneck[:, :, action] = static_free[:, :, action] & (degree <= 2)

    history_valid = torch.zeros(b, h, t, dtype=torch.bool)
    history_track_count = min(h, count)
    history_valid[:, :history_track_count] = True
    # Randomly remove some old observations, but keep ego history valid.
    if history_track_count > 1:
        dropout = torch.rand(
            b, history_track_count - 1, t, generator=generator
        ) < 0.1
        history_valid[:, 1:history_track_count] &= ~dropout
    history_track_current_slot = torch.full((b, h), -1, dtype=torch.long)
    history_track_current_slot[:, :history_track_count] = torch.arange(history_track_count)
    history_global_ids = torch.full((b, h), -1, dtype=torch.long)
    history_global_ids[:, :history_track_count] = torch.arange(history_track_count)

    base_xy = current_xy[:, :history_track_count]
    random_steps = torch.randint(
        -1, 2, (b, history_track_count, t, 2), generator=generator
    )
    cumulative = torch.cumsum(random_steps, dim=2)
    history_xy = torch.zeros(b, h, t, 2, dtype=torch.long)
    history_xy[:, :history_track_count] = base_xy[:, :, None, :] - cumulative
    history_goal_delta = torch.zeros_like(history_xy)
    history_goal_delta[:, :history_track_count] = (
        current_goal_delta[:, :history_track_count, None, :]
        + current_xy[:, :history_track_count, None, :]
        - history_xy[:, :history_track_count]
    )
    history_hops = torch.full((b, h, t), config.max_hops + 2, dtype=torch.long)
    history_hops[:, :history_track_count] = (
        history_goal_delta[:, :history_track_count].abs().sum(dim=-1)
        + torch.randint(0, 4, (b, history_track_count, t), generator=generator)
    )
    selected = torch.full((b, h, t), int(Action.PAD), dtype=torch.long)
    observed = torch.full((b, h, t), int(Action.PAD), dtype=torch.long)
    if history_track_count:
        selected[:, :history_track_count] = torch.randint(
            0, 5, (b, history_track_count, t), generator=generator
        )
        observed[:, :history_track_count] = selected[:, :history_track_count]
        mismatch = torch.rand(
            b, history_track_count, t, generator=generator
        ) < 0.15
        observed[:, :history_track_count] = torch.where(
            mismatch,
            torch.full_like(observed[:, :history_track_count], int(Action.WAIT)),
            observed[:, :history_track_count],
        )
    selected = selected.masked_fill(~history_valid, int(Action.PAD))
    observed = observed.masked_fill(~history_valid, int(Action.PAD))

    # Choose a feasible ego label.
    ego_action = torch.zeros(b, dtype=torch.long)
    for bi in range(b):
        valid_actions = torch.nonzero(static_free[bi, 0], as_tuple=False).flatten()
        choice = valid_actions[
            torch.randint(0, valid_actions.numel(), (1,), generator=generator)
        ]
        ego_action[bi] = choice

    return PolicyBatch(
        local_maps=local_maps,
        current_xy=current_xy,
        current_goal_delta=current_goal_delta,
        current_hops=current_hops,
        current_valid=current_valid,
        current_track_reset=current_track_reset,
        current_global_ids=current_global_ids,
        candidate_target_core_xy=target_core,
        candidate_in_core=in_core,
        candidate_static_free=static_free,
        candidate_one_hop_hops=one_hop,
        candidate_delta_ctg=delta_ctg,
        candidate_greedy=greedy,
        candidate_bottleneck=bottleneck,
        candidate_dynamic_occupied=torch.zeros_like(bottleneck),
        history_xy=history_xy,
        history_goal_delta=history_goal_delta,
        history_hops=history_hops,
        history_selected_action=selected,
        history_observed_move=observed,
        history_valid=history_valid,
        history_track_current_slot=history_track_current_slot,
        history_global_ids=history_global_ids,
        ego_action=ego_action,
        action_soft_targets=None,
    )


class SyntheticPolicyDataset(Dataset[PolicyBatch]):
    def __init__(self, config: ModelConfig, samples: int, seed: int = 0) -> None:
        self.config = config
        self.samples = samples
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PolicyBatch:
        batch = make_synthetic_policy_batch(
            self.config,
            batch_size=1,
            seed=self.seed + index,
            num_current_agents=1 + (index % self.config.max_current_agents),
        )
        values = {}
        for name in batch.__dataclass_fields__:
            value = getattr(batch, name)
            values[name] = value[0] if isinstance(value, torch.Tensor) else value
        return PolicyBatch(**values)


def make_synthetic_communication_group(
    config: ModelConfig,
    views: int = 6,
    *,
    seed: int = 0,
) -> tuple[PolicyBatch, CommunicationGraph]:
    batch = make_synthetic_policy_batch(
        config,
        batch_size=views,
        seed=seed,
        num_current_agents=min(views, config.max_current_agents),
    )
    neighbor_index = torch.zeros(views, config.message_neighbors, dtype=torch.long)
    neighbor_valid = torch.zeros(views, config.message_neighbors, dtype=torch.bool)
    neighbor_slot = torch.full(
        (views, config.message_neighbors), config.max_current_agents, dtype=torch.long
    )
    for receiver in range(views):
        sources = [idx for idx in range(views) if idx != receiver][: config.message_neighbors]
        for rank, source in enumerate(sources):
            neighbor_index[receiver, rank] = source
            neighbor_valid[receiver, rank] = True
            neighbor_slot[receiver, rank] = rank + 1
    return batch, CommunicationGraph(neighbor_index, neighbor_valid, neighbor_slot)
