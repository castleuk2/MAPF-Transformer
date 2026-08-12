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
from ..constants import ACTION_DELTAS, Action, DeltaCTG
from ..types import CommunicationGraph, PolicyBatch, stack_policy_batches


@dataclass(slots=True)
class Episode:
    obstacles: np.ndarray
    positions: np.ndarray
    goals: np.ndarray
    actions: np.ndarray
    distance_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    candidate_cache: dict[
        tuple[int, int], tuple[set[tuple[int, int]], set[tuple[int, int]]]
    ] = field(default_factory=dict)
    visible_cache: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict)
    degree_map: np.ndarray | None = None


class EpisodeFeatureBuilder:
    """Convert a MAPF episode into the revised ego-centered token inputs.

    Expected NPZ arrays:
      obstacles [H,W]
      positions [T+1,N,2]
      goals [N,2] or [T+1,N,2]
      actions [T,N] in WAIT,UP,DOWN,LEFT,RIGHT order
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        coordinate_order: str = "row_col",
        cache_size: int = 4,
    ) -> None:
        if coordinate_order not in {"row_col", "xy"}:
            raise ValueError("coordinate_order must be row_col or xy")
        self.config = config
        self.coordinate_order = coordinate_order
        self.cache_size = cache_size
        self._cache: OrderedDict[Path, Episode] = OrderedDict()

    def _rc(self, array: np.ndarray) -> np.ndarray:
        return array[..., ::-1].copy() if self.coordinate_order == "xy" else array.copy()

    def load_episode(self, path: str | Path) -> Episode:
        path = Path(path).resolve()
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        with np.load(path, allow_pickle=False) as data:
            episode = Episode(
                obstacles=np.asarray(data["obstacles"], dtype=np.uint8),
                positions=self._rc(np.asarray(data["positions"], dtype=np.int64)),
                goals=self._rc(np.asarray(data["goals"], dtype=np.int64)),
                actions=np.asarray(data["actions"], dtype=np.int64),
            )
            episode.degree_map = self._degree_map(episode.obstacles)
        self._cache[path] = episode
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    @staticmethod
    def _crop(obstacles: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
        radius = size // 2
        result = np.ones((size, size), dtype=np.uint8)
        h, w = obstacles.shape
        r0, c0 = int(center[0]) - radius, int(center[1]) - radius
        gr0, gc0 = max(r0, 0), max(c0, 0)
        gr1, gc1 = min(r0 + size, h), min(c0 + size, w)
        if gr0 < gr1 and gc0 < gc1:
            lr0, lc0 = gr0 - r0, gc0 - c0
            result[lr0 : lr0 + gr1 - gr0, lc0 : lc0 + gc1 - gc0] = obstacles[
                gr0:gr1, gc0:gc1
            ]
        return result

    @staticmethod
    def _degree_map(obstacles: np.ndarray) -> np.ndarray:
        free = obstacles == 0
        degree = np.zeros(obstacles.shape, dtype=np.uint8)
        degree[1:, :] += free[:-1, :]
        degree[:-1, :] += free[1:, :]
        degree[:, 1:] += free[:, :-1]
        degree[:, :-1] += free[:, 1:]
        return degree

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
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and obstacles[rr, cc] == 0
                    and dist[rr, cc] == inf
                ):
                    dist[rr, cc] = nd
                    queue.append((rr, cc))
        return dist

    def _goals_at(self, episode: Episode, step: int) -> np.ndarray:
        return episode.goals[step] if episode.goals.ndim == 3 else episode.goals

    def _distance(self, episode: Episode, goal: tuple[int, int]) -> np.ndarray:
        if goal not in episode.distance_cache:
            episode.distance_cache[goal] = self._distance_map(episode.obstacles, goal)
        return episode.distance_cache[goal]

    @staticmethod
    def _action_from_delta(delta: np.ndarray) -> int:
        value = tuple(map(int, delta.tolist()))
        try:
            return ACTION_DELTAS.index(value)
        except ValueError:
            return int(Action.UNKNOWN)

    @staticmethod
    def _degree(episode: Episode, target: tuple[int, int]) -> int:
        assert episode.degree_map is not None
        return int(episode.degree_map[target])

    def _visible(self, episode: Episode, step: int, ego: int) -> list[int]:
        key = (step, ego)
        if key in episode.visible_cache:
            return list(episode.visible_cache[key])
        radius = self.config.core_map_size // 2
        pos = episode.positions[step]
        center = pos[ego]
        relative = np.abs(pos - center)
        result = tuple(np.flatnonzero((relative[:, 0] <= radius) & (relative[:, 1] <= radius)).tolist())
        episode.visible_cache[key] = result
        return list(result)

    def _candidate_sets(
        self, episode: Episode, step: int, gid: int
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        cache_key = (step, gid)
        cached = episode.candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        position = episode.positions[step, gid]
        goal = tuple(map(int, self._goals_at(episode, step)[gid]))
        dist = self._distance(episode, goal)
        inf = np.iinfo(np.int32).max
        current_d = int(dist[tuple(position)])
        feasible: set[tuple[int, int]] = set()
        greedy: set[tuple[int, int]] = set()
        h, w = episode.obstacles.shape
        for dr, dc in ACTION_DELTAS:
            target = (int(position[0]) + dr, int(position[1]) + dc)
            if not (0 <= target[0] < h and 0 <= target[1] < w):
                continue
            if episode.obstacles[target] != 0:
                continue
            feasible.add(target)
            td = int(dist[target])
            if current_d != inf and td < current_d:
                greedy.add(target)
        if tuple(position) == goal:
            greedy.add(tuple(map(int, position)))
        result = (feasible, greedy)
        episode.candidate_cache[cache_key] = result
        return result

    def _rank_current(self, episode: Episode, step: int, ego: int) -> list[int]:
        """Select the ego followed by the nearest currently visible agents.

        Distance is Manhattan distance in the global grid.  Global agent id is
        the deterministic tie-breaker, so Python and C++ produce stable slots.
        """
        visible = [gid for gid in self._visible(episode, step, ego) if gid != ego]
        positions = episode.positions[step]
        visible.sort(
            key=lambda gid: (
                int(np.abs(positions[gid] - positions[ego]).sum()),
                gid,
            )
        )
        return [ego] + visible[: self.config.max_current_agents - 1]

    def _history_ids(
        self, episode: Episode, step: int, ego: int, current_ids: list[int]
    ) -> list[int]:
        # ``current_ids`` is already ordered by current Ego-relative distance.
        # With the revised 14-track layout this list is the full Current list:
        # Current slot i and History track i always identify the same agent.
        return current_ids[: self.config.history_tracks]

    def build(self, episode: Episode, time_step: int, ego: int) -> PolicyBatch:
        cfg = self.config
        if not (0 <= time_step < episode.actions.shape[0]):
            raise IndexError("time_step must index the actions array")
        positions = episode.positions[time_step]
        goals = self._goals_at(episode, time_step)
        current_ids = self._rank_current(episode, time_step, ego)
        history_ids = self._history_ids(episode, time_step, ego, current_ids)
        current_slot = {gid: slot for slot, gid in enumerate(current_ids)}
        center_position = positions[ego]
        center = cfg.core_map_size // 2
        local_map = self._crop(episode.obstacles, center_position, cfg.local_map_size)

        n = cfg.max_current_agents
        a = cfg.num_actions
        current_xy = np.zeros((n, 2), dtype=np.int64)
        current_goal = np.zeros((n, 2), dtype=np.int64)
        current_hops = np.full((n,), cfg.max_hops + 2, dtype=np.int64)
        current_valid = np.zeros((n,), dtype=bool)
        current_reset = np.zeros((n,), dtype=bool)
        current_global = np.full((n,), -1, dtype=np.int64)
        target_core = np.zeros((n, a, 2), dtype=np.int64)
        in_core = np.zeros((n, a), dtype=bool)
        static_free = np.zeros((n, a), dtype=bool)
        one_hop = np.full((n, a), cfg.max_hops + 1, dtype=np.int64)
        delta_ctg = np.full((n, a), int(DeltaCTG.PAD), dtype=np.int64)
        greedy = np.zeros((n, a), dtype=bool)
        bottleneck = np.zeros((n, a), dtype=bool)
        dynamic_occupied = np.zeros((n, a), dtype=bool)

        previous_ids = (
            set(self._rank_current(episode, time_step - 1, ego))
            if time_step > 0
            else {ego}
        )
        inf = np.iinfo(np.int32).max
        h_map, w_map = episode.obstacles.shape
        for slot, gid in enumerate(current_ids):
            current_valid[slot] = True
            current_global[slot] = gid
            current_reset[slot] = gid not in previous_ids
            relative = positions[gid] - center_position
            current_xy[slot] = relative
            current_goal[slot] = goals[gid] - positions[gid]
            goal = tuple(map(int, goals[gid]))
            dist = self._distance(episode, goal)
            current_d = int(dist[tuple(positions[gid])])
            current_hops[slot] = -1 if current_d == inf else current_d
            for action, (dr, dc) in enumerate(ACTION_DELTAS):
                target_global = (
                    int(positions[gid, 0]) + dr,
                    int(positions[gid, 1]) + dc,
                )
                target_core[slot, action] = relative + np.asarray((dr, dc)) + center
                in_core[slot, action] = bool(
                    0 <= target_core[slot, action, 0] < cfg.core_map_size
                    and 0 <= target_core[slot, action, 1] < cfg.core_map_size
                )
                free = (
                    0 <= target_global[0] < h_map
                    and 0 <= target_global[1] < w_map
                    and episode.obstacles[target_global] == 0
                )
                static_free[slot, action] = free
                if not free:
                    delta_ctg[slot, action] = int(DeltaCTG.BLOCKED)
                    continue
                td = int(dist[target_global])
                one_hop[slot, action] = -1 if td == inf else td
                if current_d == inf or td == inf:
                    delta_ctg[slot, action] = int(DeltaCTG.UNREACHABLE)
                elif td < current_d:
                    delta_ctg[slot, action] = int(DeltaCTG.DECREASE)
                    greedy[slot, action] = True
                elif td > current_d:
                    delta_ctg[slot, action] = int(DeltaCTG.INCREASE)
                else:
                    delta_ctg[slot, action] = int(DeltaCTG.SAME)
                bottleneck[slot, action] = self._degree(episode, target_global) <= 2
                dynamic_occupied[slot, action] = any(
                    other != gid
                    and tuple(map(int, positions[other])) == target_global
                    for other in range(positions.shape[0])
                )
            if current_hops[slot] == 0:
                greedy[slot, int(Action.WAIT)] = True

        htracks = cfg.history_tracks
        tsteps = cfg.history_steps
        history_xy = np.zeros((htracks, tsteps, 2), dtype=np.int64)
        history_goal = np.zeros((htracks, tsteps, 2), dtype=np.int64)
        history_hops = np.full((htracks, tsteps), cfg.max_hops + 2, dtype=np.int64)
        history_selected = np.full((htracks, tsteps), int(Action.PAD), dtype=np.int64)
        history_observed = np.full((htracks, tsteps), int(Action.PAD), dtype=np.int64)
        history_valid = np.zeros((htracks, tsteps), dtype=bool)
        history_current_slot = np.full((htracks,), -1, dtype=np.int64)
        history_global = np.full((htracks,), -1, dtype=np.int64)

        for track, gid in enumerate(history_ids):
            history_global[track] = gid
            history_current_slot[track] = current_slot.get(gid, -1)
            for lag in range(1, tsteps + 1):
                tau = time_step - lag
                if tau < 0:
                    continue
                # The local policy may remember only agents that were observable
                # from its own ego view at that past time.
                if gid not in self._visible(episode, tau, ego):
                    continue
                history_valid[track, lag - 1] = True
                history_xy[track, lag - 1] = (
                    episode.positions[tau, gid] - center_position
                )
                past_goals = self._goals_at(episode, tau)
                history_goal[track, lag - 1] = (
                    past_goals[gid] - episode.positions[tau, gid]
                )
                dist = self._distance(episode, tuple(map(int, past_goals[gid])))
                hd = int(dist[tuple(episode.positions[tau, gid])])
                history_hops[track, lag - 1] = -1 if hd == inf else hd
                history_selected[track, lag - 1] = int(episode.actions[tau, gid])
                movement = episode.positions[tau + 1, gid] - episode.positions[tau, gid]
                history_observed[track, lag - 1] = self._action_from_delta(movement)

        return PolicyBatch(
            local_maps=torch.from_numpy(local_map).long(),
            current_xy=torch.from_numpy(current_xy).long(),
            current_goal_delta=torch.from_numpy(current_goal).long(),
            current_hops=torch.from_numpy(current_hops).long(),
            current_valid=torch.from_numpy(current_valid),
            current_track_reset=torch.from_numpy(current_reset),
            current_global_ids=torch.from_numpy(current_global).long(),
            candidate_target_core_xy=torch.from_numpy(target_core).long(),
            candidate_in_core=torch.from_numpy(in_core),
            candidate_static_free=torch.from_numpy(static_free),
            candidate_one_hop_hops=torch.from_numpy(one_hop).long(),
            candidate_delta_ctg=torch.from_numpy(delta_ctg).long(),
            candidate_greedy=torch.from_numpy(greedy),
            candidate_bottleneck=torch.from_numpy(bottleneck),
            candidate_dynamic_occupied=torch.from_numpy(dynamic_occupied),
            history_xy=torch.from_numpy(history_xy).long(),
            history_goal_delta=torch.from_numpy(history_goal).long(),
            history_hops=torch.from_numpy(history_hops).long(),
            history_selected_action=torch.from_numpy(history_selected).long(),
            history_observed_move=torch.from_numpy(history_observed).long(),
            history_valid=torch.from_numpy(history_valid),
            history_track_current_slot=torch.from_numpy(history_current_slot).long(),
            history_global_ids=torch.from_numpy(history_global).long(),
            ego_action=torch.tensor(int(episode.actions[time_step, ego]), dtype=torch.long),
            action_soft_targets=None,
        )

    def build_all_views(
        self, episode: Episode, time_step: int
    ) -> tuple[PolicyBatch, CommunicationGraph]:
        samples = [self.build(episode, time_step, ego) for ego in range(episode.positions.shape[1])]
        batch = stack_policy_batches(samples)
        views = len(samples)
        neighbor_index = torch.zeros(
            views, self.config.message_neighbors, dtype=torch.long
        )
        neighbor_valid = torch.zeros(
            views, self.config.message_neighbors, dtype=torch.bool
        )
        neighbor_slot = torch.full(
            (views, self.config.message_neighbors),
            self.config.max_current_agents,
            dtype=torch.long,
        )
        for receiver, sample in enumerate(samples):
            rank = 0
            for track in range(1, self.config.history_tracks):
                gid = int(sample.history_global_ids[track])
                slot = int(sample.history_track_current_slot[track])
                if gid < 0 or slot < 0 or gid >= views:
                    continue
                neighbor_index[receiver, rank] = gid
                neighbor_valid[receiver, rank] = True
                neighbor_slot[receiver, rank] = slot
                rank += 1
                if rank == self.config.message_neighbors:
                    break
        return batch, CommunicationGraph(neighbor_index, neighbor_valid, neighbor_slot)


def _read_manifest(path: str | Path) -> list[Path]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        values = json.loads(text)
        if not isinstance(values, list):
            raise ValueError("JSON manifest must be a list of NPZ paths")
        return [(path.parent / item).resolve() for item in values]
    result: list[Path] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
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
            candidate = path.parent / candidate
        result.append(candidate.resolve())
    return result


class EpisodeSequenceViewDataset(Dataset[PolicyBatch]):
    """Exhaustive policy-exposure indexing used by the prior comparisons."""

    def __init__(
        self,
        manifest: str | Path,
        builder: EpisodeFeatureBuilder,
        *,
        goal_wait_keep_ratio: float = 0.2,
        max_samples: int | None = None,
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
            episode_path = Path(record["path"])
            if not episode_path.is_absolute():
                episode_path = manifest_path.parent / episode_path
            arrivals = np.asarray(record["arrival_steps"], dtype=np.int64)
            time_steps = int(record["time_steps"])
            suffix = np.maximum(0, time_steps - arrivals)
            kept_waits = np.where(
                suffix > 0,
                np.maximum(1, np.ceil(suffix * goal_wait_keep_ratio)).astype(np.int64),
                0,
            )
            sample_counts = arrivals + kept_waits
            self.records.append({
                "path": episode_path.resolve(),
                "arrivals": arrivals,
                "time_steps": time_steps,
                "sample_counts": sample_counts,
                "map_family": str(record.get("map_family", "unknown")),
                "num_agents": int(record.get("num_agents", len(arrivals))),
            })
            counts.append(int(sample_counts.sum()))
        if not self.records:
            raise ValueError(f"manifest contains no episodes: {manifest_path}")
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        total = int(self.cumulative[-1])
        self.length = total if max_samples is None else min(total, int(max_samples))
        self.goal_wait_keep_ratio = float(goal_wait_keep_ratio)
        self.builder = builder

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
            waits = self._wait_timesteps(arrival, record["time_steps"])
            time_step = int(waits[sample_index - arrival])
        episode = self.builder.load_episode(record["path"])
        return self.builder.build(episode, time_step, ego)


class RandomEpisodeViewDataset(Dataset[PolicyBatch]):
    def __init__(
        self,
        manifest: str | Path,
        builder: EpisodeFeatureBuilder,
        *,
        samples_per_epoch: int,
        seed: int = 0,
    ) -> None:
        self.paths = _read_manifest(manifest)
        if not self.paths:
            raise ValueError("manifest contains no episodes")
        self.builder = builder
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> PolicyBatch:
        rng = random.Random(self.seed + index)
        path = self.paths[rng.randrange(len(self.paths))]
        episode = self.builder.load_episode(path)
        time_step = rng.randrange(episode.actions.shape[0])
        ego = rng.randrange(episode.positions.shape[1])
        return self.builder.build(episode, time_step, ego)
