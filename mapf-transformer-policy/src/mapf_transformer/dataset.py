from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ModelConfig
from .features import (
    FrameFeatures,
    GoalDistanceCache,
    build_frame_features,
    empty_frame_features,
)
from .tracking import StableNeighborTracker


@dataclass(slots=True)
class EpisodeData:
    obstacles: np.ndarray
    positions: np.ndarray
    goals: np.ndarray
    actions: np.ndarray
    arrival_steps: np.ndarray | None = None
    neighbor_ids: np.ndarray | None = None
    neighbor_valid: np.ndarray | None = None
    track_reset: np.ndarray | None = None
    neighbor_ids_24: np.ndarray | None = None
    neighbor_valid_24: np.ndarray | None = None
    track_reset_24: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    @property
    def time_steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.actions.shape[1])

    def goals_at(self, frame: int) -> np.ndarray:
        if self.goals.ndim == 2:
            return self.goals
        if self.goals.ndim == 3:
            return self.goals[frame]
        raise ValueError("goals must have shape [N,2] or [T+1,N,2]")

    def get_arrival_steps(self) -> np.ndarray:
        if self.arrival_steps is not None:
            return np.asarray(self.arrival_steps, dtype=np.int32)
        final_goals = self.goals if self.goals.ndim == 2 else self.goals[-1]
        arrivals = np.zeros(self.num_agents, dtype=np.int32)
        for agent_id in range(self.num_agents):
            not_at_goal = np.any(self.positions[:, agent_id] != final_goals[agent_id], axis=-1)
            indices = np.flatnonzero(not_at_goal)
            arrivals[agent_id] = min(int(indices[-1] + 1), self.time_steps) if indices.size else 0
        return arrivals

    def validate(self) -> None:
        if self.obstacles.ndim != 2:
            raise ValueError("obstacles must have shape [H,W]")
        if self.positions.ndim != 3 or self.positions.shape[-1] != 2:
            raise ValueError("positions must have shape [T+1,N,2]")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape [T,N]")
        if self.positions.shape[0] != self.actions.shape[0] + 1:
            raise ValueError("positions must contain one more frame than actions")
        if self.positions.shape[1] != self.actions.shape[1]:
            raise ValueError("agent dimension mismatch")
        if self.goals.ndim == 2 and self.goals.shape != self.positions.shape[1:]:
            raise ValueError("static goals must have shape [N,2]")
        if self.goals.ndim == 3 and self.goals.shape != self.positions.shape:
            raise ValueError("dynamic goals must have shape [T+1,N,2]")
        if np.any((self.actions < 0) | (self.actions > 4)):
            raise ValueError("actions must use POGEMA order 0..4")
        arrivals = self.get_arrival_steps()
        if arrivals.shape != (self.num_agents,):
            raise ValueError("arrival_steps must have shape [N]")
        if np.any((arrivals < 0) | (arrivals > self.time_steps)):
            raise ValueError("arrival_steps must be between 0 and time_steps")
        expected_tracking24 = (self.positions.shape[0], self.num_agents, 24)
        for name, value in (
            ("neighbor_ids_24", self.neighbor_ids_24),
            ("neighbor_valid_24", self.neighbor_valid_24),
            ("track_reset_24", self.track_reset_24),
        ):
            if value is not None and value.shape != expected_tracking24:
                raise ValueError(f"{name} must have shape {expected_tracking24}")


def save_episode(path: str | Path, episode: EpisodeData, compress: bool = True) -> None:
    episode.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "obstacles": np.asarray(episode.obstacles, dtype=np.uint8),
        "positions": np.asarray(episode.positions, dtype=np.int16),
        "goals": np.asarray(episode.goals, dtype=np.int16),
        "actions": np.asarray(episode.actions, dtype=np.uint8),
        "arrival_steps": np.asarray(episode.get_arrival_steps(), dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(episode.metadata or {}, ensure_ascii=False)),
    }
    if episode.neighbor_ids is not None:
        arrays["neighbor_ids"] = np.asarray(episode.neighbor_ids, dtype=np.int16)
    if episode.neighbor_valid is not None:
        arrays["neighbor_valid"] = np.asarray(episode.neighbor_valid, dtype=bool)
    if episode.track_reset is not None:
        arrays["track_reset"] = np.asarray(episode.track_reset, dtype=bool)
    if episode.neighbor_ids_24 is not None:
        arrays["neighbor_ids_24"] = np.asarray(episode.neighbor_ids_24, dtype=np.int16)
    if episode.neighbor_valid_24 is not None:
        arrays["neighbor_valid_24"] = np.asarray(episode.neighbor_valid_24, dtype=bool)
    if episode.track_reset_24 is not None:
        arrays["track_reset_24"] = np.asarray(episode.track_reset_24, dtype=bool)
    saver = np.savez_compressed if compress else np.savez
    saver(path, **arrays)


def load_episode(path: str | Path) -> EpisodeData:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata_raw = str(data["metadata_json"].item()) if "metadata_json" in data else "{}"
        episode = EpisodeData(
            obstacles=np.asarray(data["obstacles"], dtype=np.uint8),
            positions=np.asarray(data["positions"], dtype=np.int16),
            goals=np.asarray(data["goals"], dtype=np.int16),
            actions=np.asarray(data["actions"], dtype=np.uint8),
            arrival_steps=np.asarray(data["arrival_steps"], dtype=np.int32)
            if "arrival_steps" in data
            else None,
            neighbor_ids=np.asarray(data["neighbor_ids"], dtype=np.int16)
            if "neighbor_ids" in data
            else None,
            neighbor_valid=np.asarray(data["neighbor_valid"], dtype=bool)
            if "neighbor_valid" in data
            else None,
            track_reset=np.asarray(data["track_reset"], dtype=bool)
            if "track_reset" in data
            else None,
            neighbor_ids_24=np.asarray(data["neighbor_ids_24"], dtype=np.int16)
            if "neighbor_ids_24" in data
            else None,
            neighbor_valid_24=np.asarray(data["neighbor_valid_24"], dtype=bool)
            if "neighbor_valid_24" in data
            else None,
            track_reset_24=np.asarray(data["track_reset_24"], dtype=bool)
            if "track_reset_24" in data
            else None,
            metadata=json.loads(metadata_raw),
        )
    episode.validate()
    return episode


def write_manifest(paths: Sequence[str | Path], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for path in paths:
            path = Path(path)
            try:
                stored_path = path.relative_to(manifest_path.parent)
            except ValueError:
                stored_path = path
            episode = load_episode(path)
            arrivals = episode.get_arrival_steps()
            record = {
                "path": str(stored_path),
                "time_steps": episode.time_steps,
                "num_agents": episode.num_agents,
                "arrival_steps": arrivals.tolist(),
                "num_samples": int(arrivals.sum()),
            }
            stream.write(json.dumps(record) + "\n")


def read_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "path" not in record:
                raise ValueError(f"Manifest line {line_number} has no path")
            path = Path(record["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            record["path"] = str(path.resolve())
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


class _EpisodeLRU:
    def __init__(self, max_items: int = 4) -> None:
        self.max_items = int(max_items)
        self.cache: OrderedDict[str, EpisodeData] = OrderedDict()

    def get(self, path: str) -> EpisodeData:
        if path in self.cache:
            episode = self.cache.pop(path)
            self.cache[path] = episode
            return episode
        episode = load_episode(path)
        self.cache[path] = episode
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return episode


class SequenceSampleBuilder:
    """Builds one 15-frame policy sample from a raw episode trajectory."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.config.validate()

    def _slot_data(
        self,
        episode: EpisodeData,
        frame: int,
        ego_id: int,
        tracker: StableNeighborTracker,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self.config.max_neighbors == 24 and episode.neighbor_ids_24 is not None:
            ids = episode.neighbor_ids_24[frame, ego_id]
            valid = (
                episode.neighbor_valid_24[frame, ego_id]
                if episode.neighbor_valid_24 is not None
                else ids >= 0
            )
            reset = (
                episode.track_reset_24[frame, ego_id]
                if episode.track_reset_24 is not None
                else np.zeros(24, dtype=bool)
            )
            return ids, valid, reset
        if (
            episode.neighbor_ids is not None
            and episode.neighbor_ids.shape[-1] == self.config.max_neighbors
        ):
            ids = episode.neighbor_ids[frame, ego_id]
            valid = (
                episode.neighbor_valid[frame, ego_id]
                if episode.neighbor_valid is not None
                else ids >= 0
            )
            reset = (
                episode.track_reset[frame, ego_id]
                if episode.track_reset is not None
                else np.zeros(self.config.max_neighbors, dtype=bool)
            )
            return ids, valid, reset
        # Old trajectories can contain precomputed slots with another width
        # (e.g. 15). Recompute 24 stable slots from the stored full positions.
        return None, None, None

    def build(
        self,
        episode: EpisodeData,
        ego_id: int,
        time_step: int,
        keep_history: int | None = None,
        distance_cache: GoalDistanceCache | None = None,
    ) -> dict[str, torch.Tensor]:
        episode.validate()
        if not 0 <= time_step < episode.time_steps:
            raise IndexError("time_step must index an expert action")
        if not 0 <= ego_id < episode.num_agents:
            raise IndexError("ego_id out of range")

        cfg = self.config
        first_frame = max(0, time_step - cfg.history_frames + 1)
        valid_frames = list(range(first_frame, time_step + 1))
        if keep_history is not None:
            keep_history = max(1, min(int(keep_history), len(valid_frames)))
            valid_frames = valid_frames[-keep_history:]

        pad_count = cfg.history_frames - len(valid_frames)
        frames: list[FrameFeatures] = [empty_frame_features(cfg) for _ in range(pad_count)]
        frame_valid = [False] * pad_count
        distance_cache = distance_cache or GoalDistanceCache(episode.obstacles)
        tracker = StableNeighborTracker(cfg.max_neighbors, grace_steps=1)

        # When slots were not precomputed, replay only the visible history window.
        for frame in valid_frames:
            slot_ids, slot_valid, slot_reset = self._slot_data(episode, frame, ego_id, tracker)
            previous_positions = episode.positions[frame - 1] if frame > 0 else None
            previous_action = int(episode.actions[frame - 1, ego_id]) if frame > 0 else None
            features = build_frame_features(
                obstacles=episode.obstacles,
                positions=episode.positions[frame],
                goals=episode.goals_at(frame),
                ego_id=ego_id,
                config=cfg,
                distance_cache=distance_cache,
                tracker=tracker,
                neighbor_slot_ids=slot_ids,
                neighbor_valid=slot_valid,
                neighbor_reset=slot_reset,
                previous_positions=previous_positions,
                previous_action=previous_action,
            )
            frames.append(features)
            frame_valid.append(True)

        def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(np.stack([getattr(frame, name) for frame in frames]), dtype=dtype)

        sample = {
            "local_maps": stack("local_map", torch.long),
            "agent_x": stack("agent_x", torch.long),
            "agent_y": stack("agent_y", torch.long),
            "distance": stack("distance", torch.long),
            "one_hop_ctg": stack("one_hop_ctg", torch.long),
            "agent_valid": stack("agent_valid", torch.bool),
            "track_reset": stack("track_reset", torch.bool),
            "previous_action": torch.as_tensor(
                [frame.previous_action for frame in frames], dtype=torch.long
            ),
            "actual_move": torch.as_tensor([frame.actual_move for frame in frames], dtype=torch.long),
            "outcome": torch.as_tensor([frame.outcome for frame in frames], dtype=torch.long),
            "visible_count": torch.as_tensor(
                [frame.visible_count for frame in frames], dtype=torch.long
            ),
            "frame_valid": torch.as_tensor(frame_valid, dtype=torch.bool),
            "target": torch.as_tensor(int(episode.actions[time_step, ego_id]), dtype=torch.long),
            "ego_id": torch.as_tensor(ego_id, dtype=torch.long),
            "time_step": torch.as_tensor(time_step, dtype=torch.long),
        }
        return sample


class EpisodeSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: str | Path,
        config: ModelConfig,
        history_augmentation: bool = False,
        min_history_frames: int = 1,
        goal_wait_keep_ratio: float = 0.0,
        max_samples: int | None = None,
        episode_cache_size: int = 4,
        seed: int = 42,
    ) -> None:
        self.records = read_manifest(manifest_path)
        self.config = config
        self.builder = SequenceSampleBuilder(config)
        self.history_augmentation = bool(history_augmentation)
        self.min_history_frames = int(min_history_frames)
        self.goal_wait_keep_ratio = float(goal_wait_keep_ratio)
        if not 0.0 <= self.goal_wait_keep_ratio <= 1.0:
            raise ValueError("goal_wait_keep_ratio must be in [0, 1]")
        self.max_samples = max_samples
        self.seed = int(seed)
        self._cache = _EpisodeLRU(episode_cache_size)
        self._distance_caches: OrderedDict[str, GoalDistanceCache] = OrderedDict()
        self._distance_cache_size = int(episode_cache_size)

        counts: list[int] = []
        for record in self.records:
            if "arrival_steps" not in record:
                episode = load_episode(record["path"])
                record["time_steps"] = episode.time_steps
                record["num_agents"] = episode.num_agents
                record["arrival_steps"] = episode.get_arrival_steps().tolist()
            arrivals = np.asarray(record["arrival_steps"], dtype=np.int64)
            if arrivals.shape != (int(record["num_agents"]),):
                raise ValueError(f"Invalid arrival_steps in manifest: {record['path']}")
            time_steps = int(record["time_steps"])
            sample_counts = arrivals + np.asarray(
                [
                    len(self._goal_wait_timesteps(int(arrival), time_steps))
                    for arrival in arrivals
                ],
                dtype=np.int64,
            )
            record["sample_counts"] = sample_counts.tolist()
            record["num_samples"] = int(sample_counts.sum())
            counts.append(int(record["num_samples"]))
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        total = int(self.cumulative[-1])
        self.length = min(total, int(max_samples)) if max_samples is not None else total

    def __len__(self) -> int:
        return self.length

    def _goal_wait_timesteps(self, arrival: int, time_steps: int) -> np.ndarray:
        suffix_length = max(0, int(time_steps) - int(arrival))
        if suffix_length == 0 or self.goal_wait_keep_ratio <= 0.0:
            return np.empty(0, dtype=np.int64)
        keep = min(
            suffix_length,
            max(1, int(np.ceil(suffix_length * self.goal_wait_keep_ratio))),
        )
        # Spread retained targets across the suffix. The complete sequence is
        # still present in the NPZ and therefore in every temporal history.
        offsets = np.linspace(0, suffix_length - 1, num=keep, dtype=np.int64)
        return int(arrival) + offsets

    def _resolve_index(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = int(self.cumulative[episode_index - 1]) if episode_index > 0 else 0
        local_index = index - previous
        record = self.records[episode_index]
        arrivals = np.asarray(record["arrival_steps"], dtype=np.int64)
        sample_counts = np.asarray(record["sample_counts"], dtype=np.int64)
        agent_cumulative = np.cumsum(sample_counts, dtype=np.int64)
        ego_id = int(np.searchsorted(agent_cumulative, local_index, side="right"))
        agent_previous = int(agent_cumulative[ego_id - 1]) if ego_id > 0 else 0
        agent_sample_index = int(local_index - agent_previous)
        arrival = int(arrivals[ego_id])
        if agent_sample_index < arrival:
            time_step = agent_sample_index
        else:
            wait_steps = self._goal_wait_timesteps(arrival, int(record["time_steps"]))
            time_step = int(wait_steps[agent_sample_index - arrival])
        return episode_index, int(time_step), int(ego_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, time_step, ego_id = self._resolve_index(index)
        path = self.records[episode_index]["path"]
        episode = self._cache.get(path)
        if path in self._distance_caches:
            distance_cache = self._distance_caches.pop(path)
            self._distance_caches[path] = distance_cache
        else:
            distance_cache = GoalDistanceCache(episode.obstacles)
            self._distance_caches[path] = distance_cache
            while len(self._distance_caches) > self._distance_cache_size:
                self._distance_caches.popitem(last=False)
        keep_history = None
        if self.history_augmentation:
            available = min(self.config.history_frames, time_step + 1)
            low = min(self.min_history_frames, available)
            # Deterministic per sample and worker-safe; varies if the dataset seed changes per run.
            rng = np.random.default_rng(self.seed + index * 104729)
            keep_history = int(rng.integers(low, available + 1))
        return self.builder.build(
            episode,
            ego_id,
            time_step,
            keep_history=keep_history,
            distance_cache=distance_cache,
        )


def iterate_episode_samples(
    episode: EpisodeData,
    config: ModelConfig,
) -> Iterator[dict[str, torch.Tensor]]:
    builder = SequenceSampleBuilder(config)
    for time_step in range(episode.time_steps):
        for ego_id in range(episode.num_agents):
            yield builder.build(episode, ego_id, time_step)
