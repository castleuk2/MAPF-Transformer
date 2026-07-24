from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import numpy as np

from .config import ModelConfig
from .geometry import (
    INF_DISTANCE,
    WAIT,
    action_from_displacement,
    bfs_distance_map,
    crop_local_map,
    encode_goal_distance,
    global_to_local,
    one_hop_cost_to_go,
    CTG_UNREACHABLE,
)
from .tracking import StableNeighborTracker, TrackingResult

START_ACTION = 5


class TransitionOutcome(IntEnum):
    START = 0
    SUCCESS = 1
    FAILED = 2
    WAIT = 3


@dataclass(slots=True)
class FrameFeatures:
    local_map: np.ndarray
    agent_x: np.ndarray
    agent_y: np.ndarray
    distance: np.ndarray
    one_hop_ctg: np.ndarray
    agent_valid: np.ndarray
    track_reset: np.ndarray
    previous_action: int
    actual_move: int
    outcome: int
    visible_count: int


class GoalDistanceCache:
    """Caches static distance fields for repeated agent goals on one map."""

    def __init__(self, obstacles: np.ndarray) -> None:
        self.obstacles = np.asarray(obstacles, dtype=np.uint8)
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def get(self, goal_xy: Iterable[int]) -> np.ndarray:
        key = tuple(int(v) for v in goal_xy)
        if key not in self._cache:
            self._cache[key] = bfs_distance_map(self.obstacles, key)
        return self._cache[key]


def _transition_fields(
    current_position: np.ndarray,
    previous_position: np.ndarray | None,
    previous_action: int | None,
) -> tuple[int, int, int]:
    if previous_position is None or previous_action is None:
        return START_ACTION, WAIT, int(TransitionOutcome.START)
    displacement = np.asarray(current_position, dtype=np.int16) - np.asarray(previous_position, dtype=np.int16)
    actual_move = action_from_displacement(displacement)
    previous_action = int(previous_action)
    if previous_action == WAIT:
        outcome = TransitionOutcome.WAIT
    elif actual_move == previous_action:
        outcome = TransitionOutcome.SUCCESS
    else:
        outcome = TransitionOutcome.FAILED
    return previous_action, actual_move, int(outcome)


def visible_neighbor_ids(
    positions: np.ndarray,
    ego_id: int,
    radius: int,
) -> tuple[list[int], dict[int, float]]:
    ego_xy = positions[ego_id]
    visible: list[int] = []
    scores: dict[int, float] = {}
    for agent_id, xy in enumerate(positions):
        if agent_id == ego_id:
            continue
        delta = np.asarray(xy, dtype=np.int32) - np.asarray(ego_xy, dtype=np.int32)
        if int(np.max(np.abs(delta))) <= radius:
            visible.append(agent_id)
            # Stable and deterministic; lower is more relevant.
            scores[agent_id] = float(np.abs(delta).sum())
    return visible, scores


def build_frame_features(
    obstacles: np.ndarray,
    positions: np.ndarray,
    goals: np.ndarray,
    ego_id: int,
    config: ModelConfig,
    distance_cache: GoalDistanceCache | None = None,
    tracker: StableNeighborTracker | None = None,
    neighbor_slot_ids: Sequence[int] | None = None,
    neighbor_valid: Sequence[bool] | None = None,
    neighbor_reset: Sequence[bool] | None = None,
    previous_positions: np.ndarray | None = None,
    previous_action: int | None = None,
    local_map: np.ndarray | None = None,
) -> FrameFeatures:
    """Builds one current-time frame without using future state."""
    config.validate()
    obstacles = np.asarray(obstacles, dtype=np.uint8)
    positions = np.asarray(positions, dtype=np.int16)
    goals = np.asarray(goals, dtype=np.int16)
    ego_id = int(ego_id)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_agents, 2]")
    if goals.shape != positions.shape:
        raise ValueError("goals must have the same shape as positions")
    if not 0 <= ego_id < positions.shape[0]:
        raise IndexError("ego_id out of range")

    ego_xy = positions[ego_id]
    distance_cache = distance_cache or GoalDistanceCache(obstacles)
    if local_map is None:
        local_map = crop_local_map(obstacles, ego_xy, config.map_size)
    else:
        local_map = np.asarray(local_map, dtype=np.uint8)
        if local_map.shape != (config.map_size, config.map_size):
            raise ValueError("local_map has an unexpected shape")

    if neighbor_slot_ids is None:
        visible, scores = visible_neighbor_ids(positions, ego_id, config.local_radius)
        if tracker is None:
            tracker = StableNeighborTracker(config.max_neighbors, grace_steps=0)
        tracking: TrackingResult = tracker.update(visible, scores)
        slot_ids = tracking.agent_ids
        slot_valid = tracking.valid
        slot_reset = tracking.reset
    else:
        slot_ids = np.asarray(neighbor_slot_ids, dtype=np.int32)
        if slot_ids.shape != (config.max_neighbors,):
            raise ValueError("neighbor_slot_ids has an unexpected shape")
        if neighbor_valid is None:
            slot_valid = slot_ids >= 0
        else:
            slot_valid = np.asarray(neighbor_valid, dtype=bool)
        if neighbor_reset is None:
            slot_reset = np.zeros(config.max_neighbors, dtype=bool)
        else:
            slot_reset = np.asarray(neighbor_reset, dtype=bool)

    slots = config.agents_per_frame
    agent_x = np.full(slots, 15, dtype=np.int64)
    agent_y = np.full(slots, 15, dtype=np.int64)
    distance = np.full(slots, config.distance_buckets - 1, dtype=np.int64)
    one_hop_ctg = np.full((slots, 5), CTG_UNREACHABLE, dtype=np.int64)
    valid = np.zeros(slots, dtype=bool)
    track_reset = np.zeros(slots, dtype=bool)

    for slot in range(config.max_neighbors):
        agent_id = int(slot_ids[slot])
        if agent_id < 0 or not bool(slot_valid[slot]) or agent_id >= positions.shape[0]:
            continue
        lx, ly, is_local = global_to_local(positions[agent_id], ego_xy, config.map_size)
        if not is_local:
            continue
        agent_x[slot], agent_y[slot] = lx, ly
        valid[slot] = True
        track_reset[slot] = bool(slot_reset[slot])
        distance_map = distance_cache.get(goals[agent_id])
        raw_distance = int(distance_map[tuple(positions[agent_id])])
        distance[slot] = encode_goal_distance(
            raw_distance,
            config.distance_encoding,
            config.distance_buckets,
        )
        if config.one_hop_ctg:
            one_hop_ctg[slot] = one_hop_cost_to_go(
                distance_map, obstacles, positions[agent_id]
            )

    ego_slot = config.ego_slot
    agent_x[ego_slot] = config.local_radius
    agent_y[ego_slot] = config.local_radius
    valid[ego_slot] = True
    ego_distance_map = distance_cache.get(goals[ego_id])
    raw_ego_distance = int(ego_distance_map[tuple(ego_xy)])
    distance[ego_slot] = encode_goal_distance(
        raw_ego_distance,
        config.distance_encoding,
        config.distance_buckets,
    )
    if config.one_hop_ctg:
        one_hop_ctg[ego_slot] = one_hop_cost_to_go(ego_distance_map, obstacles, ego_xy)

    prev_position = None if previous_positions is None else np.asarray(previous_positions)[ego_id]
    previous_action_value, actual_move, outcome = _transition_fields(
        current_position=ego_xy,
        previous_position=prev_position,
        previous_action=previous_action,
    )

    return FrameFeatures(
        local_map=local_map,
        agent_x=agent_x,
        agent_y=agent_y,
        distance=distance,
        one_hop_ctg=one_hop_ctg,
        agent_valid=valid,
        track_reset=track_reset,
        previous_action=previous_action_value,
        actual_move=actual_move,
        outcome=outcome,
        visible_count=int(valid.sum()),
    )


def empty_frame_features(config: ModelConfig) -> FrameFeatures:
    slots = config.agents_per_frame
    return FrameFeatures(
        local_map=np.zeros((config.map_size, config.map_size), dtype=np.uint8),
        agent_x=np.full(slots, 15, dtype=np.int64),
        agent_y=np.full(slots, 15, dtype=np.int64),
        distance=np.full(slots, config.distance_buckets - 1, dtype=np.int64),
        one_hop_ctg=np.full((slots, 5), CTG_UNREACHABLE, dtype=np.int64),
        agent_valid=np.zeros(slots, dtype=bool),
        track_reset=np.zeros(slots, dtype=bool),
        previous_action=START_ACTION,
        actual_move=WAIT,
        outcome=int(TransitionOutcome.START),
        visible_count=0,
    )
