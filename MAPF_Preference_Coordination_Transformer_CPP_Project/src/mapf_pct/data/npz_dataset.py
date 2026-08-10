from __future__ import annotations

import json
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ModelConfig
from ..constants import ACTION_DELTAS, Action, DeltaCTG, Outcome
from ..types import PolicyBatch


@dataclass(slots=True)
class _Episode:
    obstacles: np.ndarray
    positions: np.ndarray
    goals: np.ndarray
    actions: np.ndarray
    distance_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


class EpisodeFeatureBuilder:
    """Convert the upstream NPZ episode format into one ego-recentered sample.

    Expected arrays follow the baseline repository README:
      obstacles [H,W], positions [T+1,N,2], goals [N,2] or [T+1,N,2],
      actions [T,N] in WAIT/UP/DOWN/LEFT/RIGHT order.

    Internally coordinates are row/column. Set ``coordinate_order='xy'`` when
    the episode stores x/y and must be swapped before map indexing.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        coordinate_order: str = "row_col",
        task_mode: str = "one_shot",
        target_mode: str = "stay",
        cache_size: int = 4,
    ) -> None:
        if coordinate_order not in {"row_col", "xy"}:
            raise ValueError("coordinate_order must be 'row_col' or 'xy'")
        self.config = config
        self.coordinate_order = coordinate_order
        self.task_mode_id = 0 if task_mode == "one_shot" else 1
        self.target_mode_id = 0 if target_mode == "stay" else 1
        self.cache_size = cache_size
        self._episodes: OrderedDict[Path, _Episode] = OrderedDict()

    def _rc(self, array: np.ndarray) -> np.ndarray:
        return array[..., ::-1].copy() if self.coordinate_order == "xy" else array.copy()

    def load_episode(self, path: str | Path) -> _Episode:
        path = Path(path).resolve()
        cached = self._episodes.get(path)
        if cached is not None:
            self._episodes.move_to_end(path)
            return cached
        with np.load(path, allow_pickle=False) as data:
            episode = _Episode(
                obstacles=np.asarray(data["obstacles"], dtype=np.uint8),
                positions=self._rc(np.asarray(data["positions"], dtype=np.int64)),
                goals=self._rc(np.asarray(data["goals"], dtype=np.int64)),
                actions=np.asarray(data["actions"], dtype=np.int64),
            )
        self._episodes[path] = episode
        while len(self._episodes) > self.cache_size:
            self._episodes.popitem(last=False)
        return episode

    @staticmethod
    def _crop_with_obstacle_padding(
        obstacles: np.ndarray, center: np.ndarray, size: int
    ) -> np.ndarray:
        radius = size // 2
        result = np.ones((size, size), dtype=np.uint8)
        h, w = obstacles.shape
        r0, c0 = int(center[0]) - radius, int(center[1]) - radius
        for lr in range(size):
            for lc in range(size):
                rr, cc = r0 + lr, c0 + lc
                if 0 <= rr < h and 0 <= cc < w:
                    result[lr, lc] = obstacles[rr, cc]
        return result

    @staticmethod
    def _distance_map(obstacles: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
        h, w = obstacles.shape
        inf = np.iinfo(np.int32).max
        dist = np.full((h, w), inf, dtype=np.int32)
        gr, gc = goal
        if not (0 <= gr < h and 0 <= gc < w) or obstacles[gr, gc] != 0:
            return dist
        queue: deque[tuple[int, int]] = deque([(gr, gc)])
        dist[gr, gc] = 0
        while queue:
            r, c = queue.popleft()
            nd = int(dist[r, c]) + 1
            for dr, dc in ACTION_DELTAS[1:]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and obstacles[rr, cc] == 0 and dist[rr, cc] == inf:
                    dist[rr, cc] = nd
                    queue.append((rr, cc))
        return dist

    def _get_distance(self, episode: _Episode, goal: tuple[int, int]) -> np.ndarray:
        if goal not in episode.distance_cache:
            episode.distance_cache[goal] = self._distance_map(episode.obstacles, goal)
        return episode.distance_cache[goal]

    @staticmethod
    def _action_from_delta(delta: np.ndarray) -> int:
        value = tuple(map(int, delta.tolist()))
        try:
            return ACTION_DELTAS.index(value)
        except ValueError:
            return int(Action.WAIT)

    @staticmethod
    def _degree(obstacles: np.ndarray, target: tuple[int, int]) -> int:
        h, w = obstacles.shape
        r, c = target
        degree = 0
        for dr, dc in ACTION_DELTAS[1:]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and obstacles[rr, cc] == 0:
                degree += 1
        return degree

    def build_episode(self, episode: _Episode, time_step: int, ego: int) -> PolicyBatch:
        t_max, n_global = episode.actions.shape
        if not (0 <= time_step < t_max):
            raise IndexError(f"time_step must be in [0,{t_max - 1}]")
        if not (0 <= ego < n_global):
            raise IndexError(f"ego must be in [0,{n_global - 1}]")
        cfg = self.config
        current = episode.positions[time_step]
        ego_pos = current[ego]
        radius = cfg.core_map_size // 2

        visible: list[int] = []
        for agent in range(n_global):
            delta = current[agent] - ego_pos
            if abs(int(delta[0])) <= radius and abs(int(delta[1])) <= radius:
                visible.append(agent)
        visible.sort(key=lambda idx: (0 if idx == ego else 1, int(np.abs(current[idx] - ego_pos).sum()), idx))
        slots = visible[: cfg.max_agents]
        if ego not in slots:
            slots = [ego] + slots[: cfg.max_agents - 1]
        slots = [ego] + [idx for idx in slots if idx != ego]
        valid_count = len(slots)

        local_map_np = self._crop_with_obstacle_padding(
            episode.obstacles, ego_pos, cfg.local_map_size
        )
        local_map = torch.from_numpy(local_map_np.astype(np.int64))
        pad_coord = cfg.core_map_size + 1
        agent_xy = torch.full((cfg.max_agents, 2), pad_coord, dtype=torch.long)
        goal_delta = torch.zeros(cfg.max_agents, 2, dtype=torch.long)
        remaining_hops = torch.full((cfg.max_agents,), cfg.max_hops + 2, dtype=torch.long)
        agent_valid = torch.zeros(cfg.max_agents, dtype=torch.bool)
        track_reset = torch.zeros(cfg.max_agents, dtype=torch.bool)
        on_goal = torch.zeros(cfg.max_agents, dtype=torch.bool)
        goal_outside = torch.zeros(cfg.max_agents, dtype=torch.bool)
        slot_goals: list[tuple[int, int] | None] = [None] * cfg.max_agents
        distance_maps: list[np.ndarray | None] = [None] * cfg.max_agents

        goals_t = episode.goals[time_step] if episode.goals.ndim == 3 else episode.goals
        for slot, gid in enumerate(slots):
            agent_valid[slot] = True
            local = current[gid] - ego_pos + radius
            agent_xy[slot] = torch.from_numpy(local.astype(np.int64))
            goal = tuple(map(int, goals_t[gid].tolist()))
            slot_goals[slot] = goal
            displacement = goals_t[gid] - current[gid]
            goal_delta[slot] = torch.from_numpy(
                np.clip(displacement, -cfg.goal_delta_clip, cfg.goal_delta_clip).astype(np.int64)
            )
            goal_outside[slot] = bool(np.abs(displacement).max() > radius)
            on_goal[slot] = bool(np.array_equal(current[gid], goals_t[gid]))
            dist_map = self._get_distance(episode, goal)
            distance_maps[slot] = dist_map
            value = int(dist_map[tuple(current[gid])])
            if value == np.iinfo(np.int32).max:
                remaining_hops[slot] = cfg.max_hops + 1
            else:
                remaining_hops[slot] = min(value, cfg.max_hops + 2)

        # Per-agent five-step history, aligned oldest -> newest.
        hs = cfg.history_steps
        history_selected = torch.full((cfg.max_agents, hs), int(Action.PAD), dtype=torch.long)
        history_executed = torch.full_like(history_selected, int(Action.PAD))
        history_outcome = torch.full_like(history_selected, int(Outcome.PAD))
        history_delta = torch.full_like(history_selected, int(DeltaCTG.PAD))
        history_valid = torch.zeros(cfg.max_agents, hs, dtype=torch.bool)
        for slot, gid in enumerate(slots):
            dist_map = distance_maps[slot]
            assert dist_map is not None
            for hidx, step in enumerate(range(time_step - hs, time_step)):
                if step < 0:
                    continue
                history_valid[slot, hidx] = True
                selected = int(episode.actions[step, gid])
                executed = self._action_from_delta(
                    episode.positions[step + 1, gid] - episode.positions[step, gid]
                )
                history_selected[slot, hidx] = selected
                history_executed[slot, hidx] = executed
                if selected == int(Action.WAIT) and executed == int(Action.WAIT):
                    at_goal = np.array_equal(episode.positions[step, gid], goals_t[gid])
                    history_outcome[slot, hidx] = int(Outcome.AT_GOAL if at_goal else Outcome.WAIT)
                elif selected != executed:
                    history_outcome[slot, hidx] = int(Outcome.BLOCKED_OR_OVERRIDDEN)
                else:
                    history_outcome[slot, hidx] = int(Outcome.SUCCESS)
                before = int(dist_map[tuple(episode.positions[step, gid])])
                after = int(dist_map[tuple(episode.positions[step + 1, gid])])
                inf = np.iinfo(np.int32).max
                if before == inf or after == inf:
                    history_delta[slot, hidx] = int(DeltaCTG.UNREACHABLE)
                elif after < before:
                    history_delta[slot, hidx] = int(DeltaCTG.DECREASE)
                elif after == before:
                    history_delta[slot, hidx] = int(DeltaCTG.SAME)
                else:
                    history_delta[slot, hidx] = int(DeltaCTG.INCREASE)

        # Candidate features in the ego-centered core coordinate system.
        shape = (cfg.max_agents, cfg.num_actions)
        candidate_target_xy = torch.zeros(*shape, 2, dtype=torch.long)
        candidate_in_view = torch.zeros(shape, dtype=torch.bool)
        candidate_delta_ctg = torch.full(shape, int(DeltaCTG.PAD), dtype=torch.long)
        candidate_greedy = torch.zeros(shape, dtype=torch.bool)
        candidate_static_free = torch.zeros(shape, dtype=torch.bool)
        candidate_target_occupied = torch.zeros(shape, dtype=torch.bool)
        candidate_contenders = torch.zeros(shape, dtype=torch.long)
        candidate_edge_swap = torch.zeros(shape, dtype=torch.bool)
        candidate_bottleneck = torch.zeros(shape, dtype=torch.bool)
        candidate_congestion = torch.zeros(shape, dtype=torch.float32)
        global_targets: list[list[tuple[int, int]]] = [[(0, 0)] * cfg.num_actions for _ in range(cfg.max_agents)]
        occupied = {tuple(map(int, current[gid].tolist())): slot for slot, gid in enumerate(slots)}
        map_h, map_w = episode.obstacles.shape
        inf = np.iinfo(np.int32).max

        for slot, gid in enumerate(slots):
            pos = current[gid]
            dist_map = distance_maps[slot]
            assert dist_map is not None
            current_d = int(dist_map[tuple(pos)])
            for action, (dr, dc) in enumerate(ACTION_DELTAS):
                target = (int(pos[0]) + dr, int(pos[1]) + dc)
                global_targets[slot][action] = target
                local_target = np.asarray(target) - ego_pos + radius
                in_view = bool(
                    0 <= local_target[0] < cfg.core_map_size
                    and 0 <= local_target[1] < cfg.core_map_size
                )
                candidate_in_view[slot, action] = in_view
                candidate_target_xy[slot, action] = torch.from_numpy(
                    np.clip(local_target, 0, cfg.core_map_size - 1).astype(np.int64)
                )
                inside = 0 <= target[0] < map_h and 0 <= target[1] < map_w
                free = inside and episode.obstacles[target] == 0
                candidate_static_free[slot, action] = bool(free)
                if not free:
                    candidate_delta_ctg[slot, action] = int(DeltaCTG.BLOCKED)
                    continue
                target_d = int(dist_map[target])
                if current_d == inf or target_d == inf:
                    state = DeltaCTG.UNREACHABLE
                elif target_d < current_d:
                    state = DeltaCTG.DECREASE
                elif target_d == current_d:
                    state = DeltaCTG.SAME
                else:
                    state = DeltaCTG.INCREASE
                candidate_delta_ctg[slot, action] = int(state)
                candidate_greedy[slot, action] = state == DeltaCTG.DECREASE
                candidate_target_occupied[slot, action] = target in occupied and occupied[target] != slot
                candidate_bottleneck[slot, action] = self._degree(episode.obstacles, target) <= 2
                nearby = sum(
                    1
                    for other_gid in slots
                    if other_gid != gid and int(np.abs(current[other_gid] - np.asarray(target)).sum()) <= 2
                )
                candidate_congestion[slot, action] = nearby / max(1, valid_count - 1)
            if on_goal[slot]:
                candidate_greedy[slot, int(Action.WAIT)] = True

        for i in range(valid_count):
            for action in range(cfg.num_actions):
                target = global_targets[i][action]
                contenders = 0
                for j in range(valid_count):
                    if i == j:
                        continue
                    for other_action in range(cfg.num_actions):
                        if candidate_greedy[j, other_action] and global_targets[j][other_action] == target:
                            contenders += 1
                        if (
                            global_targets[i][action] == tuple(map(int, current[slots[j]].tolist()))
                            and candidate_greedy[j, other_action]
                            and global_targets[j][other_action] == tuple(map(int, current[slots[i]].tolist()))
                        ):
                            candidate_edge_swap[i, action] = True
                candidate_contenders[i, action] = min(contenders, cfg.contender_buckets - 1)

        labels = torch.zeros(cfg.max_agents, dtype=torch.long)
        for slot, gid in enumerate(slots):
            labels[slot] = int(episode.actions[time_step, gid])

        # Heuristic reason labels are explicitly marked as such in the README.
        reasons = torch.zeros(cfg.max_agents, cfg.reason_classes)
        reason_valid = agent_valid.clone()
        for slot in range(valid_count):
            chosen = int(labels[slot])
            if candidate_delta_ctg[slot, chosen] == int(DeltaCTG.DECREASE):
                reasons[slot, 0] = 1.0
            if on_goal[slot] and chosen == int(Action.WAIT):
                reasons[slot, 1] = 1.0
            if chosen == int(Action.WAIT) and candidate_contenders[slot].max() > 0:
                reasons[slot, 5] = 1.0
            if candidate_bottleneck[slot, chosen]:
                reasons[slot, 6] = 1.0
            recent_waits = int((history_selected[slot] == int(Action.WAIT)).sum())
            if recent_waits >= 3 and chosen != int(Action.WAIT):
                reasons[slot, 9] = 1.0

        event_action = history_selected[0].clone()
        event_executed = history_executed[0].clone()
        event_outcome = history_outcome[0].clone()
        event_valid = history_valid[0].clone()
        event_numeric = torch.zeros(hs, cfg.event_numeric_dim)
        for hidx in range(hs):
            valid_agents = history_valid[:, hidx] & agent_valid
            if not valid_agents.any():
                continue
            selected = history_selected[:, hidx]
            executed = history_executed[:, hidx]
            event_numeric[hidx] = torch.tensor(
                (
                    float((executed[valid_agents] != int(Action.WAIT)).float().mean()),
                    float((selected[valid_agents] == int(Action.WAIT)).float().mean()),
                    float((selected[valid_agents] != executed[valid_agents]).float().mean()),
                    valid_count / cfg.max_agents,
                    float((history_delta[:, hidx][valid_agents] == int(DeltaCTG.DECREASE)).float().mean()),
                    float(candidate_contenders[:valid_count].max()) / max(1, cfg.contender_buckets - 1),
                    float(candidate_edge_swap[:valid_count].float().mean()),
                    float(on_goal[:valid_count].float().mean()),
                )
            )

        finite_hops = remaining_hops[:valid_count].float().clamp_max(cfg.max_hops) / max(1, cfg.max_hops)
        core = local_map[1:-1, 1:-1]
        scene_numeric = torch.tensor(
            (
                float(core.float().mean()),
                valid_count / cfg.max_agents,
                float(on_goal[:valid_count].float().mean()),
                float(finite_hops.mean()),
                float(finite_hops.std(unbiased=False)),
                float(finite_hops.min()),
                float(finite_hops.max()),
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
            2 if scene_numeric[11] + scene_numeric[12] > 0.35 else 1 if scene_numeric[11] + scene_numeric[12] > 0.12 else 0,
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
            task_mode=torch.tensor(self.task_mode_id),
            target_mode=torch.tensor(self.target_mode_id),
            all_agent_actions=labels,
            action_soft_targets=None,
            reason_labels=reasons,
            reason_valid=reason_valid,
            scene_risk_labels=scene_risk,
        )

    def build(self, path: str | Path, time_step: int, ego: int) -> PolicyBatch:
        return self.build_episode(self.load_episode(path), time_step, ego)


def _manifest_paths(path: str | Path) -> list[Path]:
    manifest = Path(path)
    result: list[Path] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = line
        if isinstance(value, str):
            episode_path = value
        elif isinstance(value, dict):
            episode_path = value.get("path") or value.get("episode") or value.get("file")
            if episode_path is None:
                raise ValueError(f"manifest line has no path field: {line}")
        else:
            raise ValueError(f"unsupported manifest line: {line}")
        candidate = Path(episode_path)
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        result.append(candidate.resolve())
    if not result:
        raise ValueError(f"manifest contains no episode paths: {manifest}")
    return result


class RandomEpisodeSampleDataset(Dataset[PolicyBatch]):
    """Deterministic random sampling over expert NPZ episodes."""

    def __init__(
        self,
        config: ModelConfig,
        manifest: str | Path,
        size: int,
        *,
        seed: int = 0,
        coordinate_order: str = "row_col",
        task_mode: str = "one_shot",
        target_mode: str = "stay",
    ) -> None:
        self.paths = _manifest_paths(manifest)
        self.size = int(size)
        self.seed = int(seed)
        self.builder = EpisodeFeatureBuilder(
            config,
            coordinate_order=coordinate_order,
            task_mode=task_mode,
            target_mode=target_mode,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> PolicyBatch:
        rng = random.Random(self.seed + int(index) * 1000003)
        path = self.paths[rng.randrange(len(self.paths))]
        episode = self.builder.load_episode(path)
        t = rng.randrange(episode.actions.shape[0])
        ego = rng.randrange(episode.actions.shape[1])
        return self.builder.build(path, t, ego)


class EpisodeSequenceSampleDataset(Dataset[PolicyBatch]):
    """Baseline-compatible policy-exposure indexing over expert episodes.

    Every ego contributes all pre-arrival time steps. A deterministic, evenly
    spaced fraction of its post-arrival WAIT suffix is retained, matching the
    original baseline dataset's ``goal_wait_keep_ratio`` behavior.
    """

    def __init__(
        self,
        config: ModelConfig,
        manifest: str | Path,
        *,
        goal_wait_keep_ratio: float = 0.2,
        max_samples: int | None = None,
        coordinate_order: str = "row_col",
        task_mode: str = "one_shot",
        target_mode: str = "stay",
    ) -> None:
        if not 0.0 <= goal_wait_keep_ratio <= 1.0:
            raise ValueError("goal_wait_keep_ratio must be in [0,1]")
        manifest_path = Path(manifest)
        self.records: list[dict] = []
        counts: list[int] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            path = Path(record["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            arrivals = np.asarray(record["arrival_steps"], dtype=np.int64)
            time_steps = int(record["time_steps"])
            suffix = np.maximum(0, time_steps - arrivals)
            kept_waits = np.where(
                suffix > 0,
                np.maximum(1, np.ceil(suffix * goal_wait_keep_ratio)).astype(np.int64),
                0,
            )
            sample_counts = arrivals + kept_waits
            self.records.append(
                {
                    "path": path.resolve(),
                    "arrivals": arrivals,
                    "time_steps": time_steps,
                    "sample_counts": sample_counts,
                    "map_family": record.get("map_family"),
                    "num_agents": int(record.get("num_agents", len(arrivals))),
                }
            )
            counts.append(int(sample_counts.sum()))
        if not self.records:
            raise ValueError(f"manifest contains no episodes: {manifest_path}")
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        total = int(self.cumulative[-1])
        self.length = total if max_samples is None else min(total, int(max_samples))
        self.goal_wait_keep_ratio = float(goal_wait_keep_ratio)
        self.builder = EpisodeFeatureBuilder(
            config,
            coordinate_order=coordinate_order,
            task_mode=task_mode,
            target_mode=target_mode,
        )

    def __len__(self) -> int:
        return self.length

    def _wait_timesteps(self, arrival: int, time_steps: int) -> np.ndarray:
        suffix = max(0, time_steps - arrival)
        if suffix == 0 or self.goal_wait_keep_ratio == 0.0:
            return np.empty(0, dtype=np.int64)
        keep = min(suffix, max(1, int(math.ceil(suffix * self.goal_wait_keep_ratio))))
        return arrival + np.linspace(0, suffix - 1, num=keep, dtype=np.int64)

    def __getitem__(self, index: int) -> PolicyBatch:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = int(self.cumulative[episode_index - 1]) if episode_index else 0
        local_index = index - previous
        record = self.records[episode_index]
        agent_cumulative = np.cumsum(record["sample_counts"], dtype=np.int64)
        ego = int(np.searchsorted(agent_cumulative, local_index, side="right"))
        agent_previous = int(agent_cumulative[ego - 1]) if ego else 0
        sample_index = int(local_index - agent_previous)
        arrival = int(record["arrivals"][ego])
        if sample_index < arrival:
            time_step = sample_index
        else:
            time_step = int(self._wait_timesteps(arrival, record["time_steps"])[sample_index - arrival])
        return self.builder.build(record["path"], time_step, ego)
