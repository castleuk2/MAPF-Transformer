from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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
    metadata: dict[str, Any] | None = None

    def get_arrival_steps(self) -> np.ndarray:
        if self.arrival_steps is not None:
            return np.asarray(self.arrival_steps, dtype=np.int32)
        final_goals = self.goals if self.goals.ndim == 2 else self.goals[-1]
        arrivals = np.zeros(self.actions.shape[1], dtype=np.int32)
        for agent_id in range(self.actions.shape[1]):
            not_at_goal = np.any(self.positions[:, agent_id] != final_goals[agent_id], axis=-1)
            indices = np.flatnonzero(not_at_goal)
            arrivals[agent_id] = min(int(indices[-1] + 1), self.actions.shape[0]) if indices.size else 0
        return arrivals

    def validate(self) -> None:
        if self.obstacles.ndim != 2:
            raise ValueError("obstacles must have shape [H,W]")
        if self.positions.ndim != 3 or self.positions.shape[-1] != 2:
            raise ValueError("positions must have shape [T+1,N,2]")
        if self.actions.ndim != 2 or self.positions.shape[:2] != (
            self.actions.shape[0] + 1,
            self.actions.shape[1],
        ):
            raise ValueError("actions [T,N] and positions [T+1,N,2] are inconsistent")
        if self.goals.ndim == 2 and self.goals.shape != self.positions.shape[1:]:
            raise ValueError("static goals must have shape [N,2]")
        if self.goals.ndim == 3 and self.goals.shape != self.positions.shape:
            raise ValueError("dynamic goals must have shape [T+1,N,2]")
        if np.any((self.actions < 0) | (self.actions > 4)):
            raise ValueError("actions must use POGEMA order 0..4")
        arrivals = self.get_arrival_steps()
        if arrivals.shape != (self.actions.shape[1],):
            raise ValueError("arrival_steps must have shape [N]")
        if np.any((arrivals < 0) | (arrivals > self.actions.shape[0])):
            raise ValueError("arrival_steps must be between 0 and makespan")


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
    (np.savez_compressed if compress else np.savez)(path, **arrays)


def write_manifest(paths: Sequence[str | Path], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for path_value in paths:
            path = Path(path_value)
            try:
                stored_path = path.relative_to(manifest_path.parent)
            except ValueError:
                stored_path = path
            with np.load(path, allow_pickle=False) as data:
                shape = data["actions"].shape
                if "arrival_steps" in data:
                    arrivals = np.asarray(data["arrival_steps"], dtype=np.int32)
                else:
                    positions = np.asarray(data["positions"], dtype=np.int16)
                    goals = np.asarray(data["goals"], dtype=np.int16)
                    final_goals = goals if goals.ndim == 2 else goals[-1]
                    arrivals = np.zeros(shape[1], dtype=np.int32)
                    for agent_id in range(shape[1]):
                        indices = np.flatnonzero(
                            np.any(positions[:, agent_id] != final_goals[agent_id], axis=-1)
                        )
                        arrivals[agent_id] = min(int(indices[-1] + 1), shape[0]) if indices.size else 0
            stream.write(json.dumps({
                "path": str(stored_path),
                "time_steps": int(shape[0]),
                "num_agents": int(shape[1]),
                "arrival_steps": arrivals.tolist(),
                "num_samples": int(arrivals.sum()),
            }) + "\n")


class StableNeighborTracker:
    def __init__(self, max_neighbors: int, grace_steps: int) -> None:
        self.max_neighbors = int(max_neighbors)
        self.grace_steps = int(grace_steps)
        self.slot_ids = np.full(self.max_neighbors, -1, dtype=np.int32)
        self.missing_steps = np.zeros(self.max_neighbors, dtype=np.int16)

    def update(self, visible_ids: Sequence[int], scores: Mapping[int, float]):
        visible = {int(agent_id) for agent_id in visible_ids}
        reset = np.zeros(self.max_neighbors, dtype=bool)
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
        assigned = {int(value) for value in self.slot_ids if value >= 0}
        candidates = sorted(visible - assigned, key=lambda agent_id: (scores.get(agent_id, 0), agent_id))
        for slot in np.flatnonzero(self.slot_ids < 0):
            if not candidates:
                break
            self.slot_ids[slot] = candidates.pop(0)
            self.missing_steps[slot] = 0
            reset[slot] = True
        valid = np.asarray([agent_id in visible for agent_id in self.slot_ids], dtype=bool)
        return self.slot_ids.copy(), valid, reset


def visible_neighbor_ids(positions: np.ndarray, ego_id: int, radius: int):
    ego = positions[ego_id]
    visible: list[int] = []
    scores: dict[int, float] = {}
    for agent_id, position in enumerate(positions):
        if agent_id == ego_id:
            continue
        delta = np.asarray(position, dtype=np.int32) - np.asarray(ego, dtype=np.int32)
        if int(np.max(np.abs(delta))) <= radius:
            visible.append(agent_id)
            scores[agent_id] = float(np.abs(delta).sum())
    return visible, scores


def precompute_neighbor_tracking(
    positions: np.ndarray,
    max_neighbors: int = 15,
    radius: int = 7,
    grace_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes online-causal stable neighbor slots for every Ego trajectory."""
    positions = np.asarray(positions, dtype=np.int16)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [T+1,N,2]")
    frames, num_agents = positions.shape[:2]
    ids = np.full((frames, num_agents, max_neighbors), -1, dtype=np.int16)
    valid = np.zeros((frames, num_agents, max_neighbors), dtype=bool)
    reset = np.zeros((frames, num_agents, max_neighbors), dtype=bool)

    trackers = [StableNeighborTracker(max_neighbors, grace_steps=grace_steps) for _ in range(num_agents)]
    for frame in range(frames):
        for ego_id in range(num_agents):
            visible, scores = visible_neighbor_ids(positions[frame], ego_id, radius)
            agent_ids, agent_valid, agent_reset = trackers[ego_id].update(visible, scores)
            ids[frame, ego_id] = agent_ids
            valid[frame, ego_id] = agent_valid
            reset[frame, ego_id] = agent_reset
    return ids, valid, reset


def build_episode_data(
    obstacles: np.ndarray,
    positions: np.ndarray,
    goals: np.ndarray,
    actions: np.ndarray,
    metadata: dict | None = None,
    precompute_tracking: bool = True,
    tracking_grace_steps: int = 1,
) -> EpisodeData:
    neighbor_ids = neighbor_valid = track_reset = None
    if precompute_tracking:
        neighbor_ids, neighbor_valid, track_reset = precompute_neighbor_tracking(
            positions,
            max_neighbors=15,
            radius=7,
            grace_steps=tracking_grace_steps,
        )
    return EpisodeData(
        obstacles=np.asarray(obstacles, dtype=np.uint8),
        positions=np.asarray(positions, dtype=np.int16),
        goals=np.asarray(goals, dtype=np.int16),
        actions=np.asarray(actions, dtype=np.uint8),
        arrival_steps=None,
        neighbor_ids=neighbor_ids,
        neighbor_valid=neighbor_valid,
        track_reset=track_reset,
        metadata=metadata or {},
    )
