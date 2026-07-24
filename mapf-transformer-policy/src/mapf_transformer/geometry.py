from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

# POGEMA-compatible action order.
WAIT, UP, DOWN, LEFT, RIGHT = range(5)
MOVES = np.asarray(
    [
        [0, 0],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ],
    dtype=np.int16,
)
MASK_ACTION_IDS = (UP, DOWN, LEFT, RIGHT)
INF_DISTANCE = np.iinfo(np.int32).max // 4

# Per-action one-hop cost-to-go categories. These describe static navigation
# progress only; dynamic agents are deliberately not treated as obstacles.
CTG_DECREASE, CTG_SAME, CTG_INCREASE, CTG_BLOCKED, CTG_UNREACHABLE = range(5)
ONE_HOP_CTG_STATES = 5


def crop_local_map(
    obstacles: np.ndarray,
    center_xy: Iterable[int],
    map_size: int = 15,
    outside_value: int = 1,
) -> np.ndarray:
    """Returns an Ego-centered square map, treating out-of-map cells as blocked."""
    obstacles = np.asarray(obstacles, dtype=np.uint8)
    if obstacles.ndim != 2:
        raise ValueError(f"obstacles must be 2-D, received {obstacles.shape}")
    if map_size <= 0 or map_size % 2 == 0:
        raise ValueError("map_size must be positive and odd")
    x, y = (int(v) for v in center_xy)
    radius = map_size // 2
    result = np.full((map_size, map_size), outside_value, dtype=np.uint8)

    src_x0 = max(0, x - radius)
    src_y0 = max(0, y - radius)
    src_x1 = min(obstacles.shape[0], x + radius + 1)
    src_y1 = min(obstacles.shape[1], y + radius + 1)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return result

    dst_x0 = src_x0 - (x - radius)
    dst_y0 = src_y0 - (y - radius)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    result[dst_x0:dst_x1, dst_y0:dst_y1] = obstacles[src_x0:src_x1, src_y0:src_y1]
    return result


def bfs_distance_map(obstacles: np.ndarray, goal_xy: Iterable[int]) -> np.ndarray:
    """Computes a static 4-connected shortest-path distance map rooted at goal."""
    obstacles = np.asarray(obstacles, dtype=np.uint8)
    if obstacles.ndim != 2:
        raise ValueError("obstacles must be 2-D")
    gx, gy = (int(v) for v in goal_xy)
    distances = np.full(obstacles.shape, INF_DISTANCE, dtype=np.int32)
    if not (0 <= gx < obstacles.shape[0] and 0 <= gy < obstacles.shape[1]):
        return distances
    if obstacles[gx, gy] != 0:
        return distances

    queue: deque[tuple[int, int]] = deque([(gx, gy)])
    distances[gx, gy] = 0
    while queue:
        x, y = queue.popleft()
        next_distance = int(distances[x, y]) + 1
        for action in MASK_ACTION_IDS:
            dx, dy = MOVES[action]
            nx, ny = x + int(dx), y + int(dy)
            if (
                0 <= nx < obstacles.shape[0]
                and 0 <= ny < obstacles.shape[1]
                and obstacles[nx, ny] == 0
                and distances[nx, ny] == INF_DISTANCE
            ):
                distances[nx, ny] = next_distance
                queue.append((nx, ny))
    return distances


def shortest_path_action_mask(
    distance_map: np.ndarray,
    position_xy: Iterable[int],
) -> np.ndarray:
    """Returns [UP, DOWN, LEFT, RIGHT] multi-hot shortest-path descent actions."""
    x, y = (int(v) for v in position_xy)
    result = np.zeros(4, dtype=np.uint8)
    if not (0 <= x < distance_map.shape[0] and 0 <= y < distance_map.shape[1]):
        return result
    current = int(distance_map[x, y])
    if current <= 0 or current >= INF_DISTANCE:
        return result
    for bit, action in enumerate(MASK_ACTION_IDS):
        dx, dy = MOVES[action]
        nx, ny = x + int(dx), y + int(dy)
        if (
            0 <= nx < distance_map.shape[0]
            and 0 <= ny < distance_map.shape[1]
            and int(distance_map[nx, ny]) == current - 1
        ):
            result[bit] = 1
    return result


def one_hop_cost_to_go(
    distance_map: np.ndarray,
    obstacles: np.ndarray,
    position_xy: Iterable[int],
) -> np.ndarray:
    """Classifies WAIT/UP/DOWN/LEFT/RIGHT by one-hop static cost-to-go change."""
    distance_map = np.asarray(distance_map)
    obstacles = np.asarray(obstacles)
    x, y = (int(v) for v in position_xy)
    result = np.full(5, CTG_UNREACHABLE, dtype=np.int64)
    if not (0 <= x < distance_map.shape[0] and 0 <= y < distance_map.shape[1]):
        return result

    current = int(distance_map[x, y])
    result[WAIT] = CTG_SAME if current < INF_DISTANCE else CTG_UNREACHABLE
    for action in MASK_ACTION_IDS:
        dx, dy = MOVES[action]
        nx, ny = x + int(dx), y + int(dy)
        if (
            not (0 <= nx < distance_map.shape[0] and 0 <= ny < distance_map.shape[1])
            or obstacles[nx, ny] != 0
        ):
            result[action] = CTG_BLOCKED
            continue
        next_distance = int(distance_map[nx, ny])
        if current >= INF_DISTANCE or next_distance >= INF_DISTANCE:
            result[action] = CTG_UNREACHABLE
        elif next_distance < current:
            result[action] = CTG_DECREASE
        elif next_distance > current:
            result[action] = CTG_INCREASE
        else:
            result[action] = CTG_SAME
    return result


def quantize_distance(distance: int) -> int:
    """Legacy 4-hop distance buckets used by existing checkpoints."""
    distance = int(distance)
    if distance <= 0:
        return 0
    if distance >= INF_DISTANCE:
        return 63
    return min(63, (distance + 3) // 4)


def encode_goal_distance(
    distance: int,
    encoding: str = "bucket4",
    num_buckets: int = 64,
) -> int:
    """Encodes goal distance while reserving the final value for unreachable.

    ``bucket4`` is the legacy 64-value representation. ``exact`` preserves
    exact shortest-path steps from 0 through ``num_buckets - 2`` and saturates
    longer reachable paths at that last reachable value. The final value is
    reserved for unreachable cells.
    """
    distance = int(distance)
    if encoding == "bucket4":
        if num_buckets != 64:
            raise ValueError("Legacy bucket4 distance encoding requires 64 buckets")
        return quantize_distance(distance)
    if encoding != "exact":
        raise ValueError(f"Unsupported goal-distance encoding: {encoding}")
    if num_buckets < 2:
        raise ValueError("Exact distance encoding requires at least 2 buckets")
    if distance >= INF_DISTANCE:
        return num_buckets - 1
    return min(num_buckets - 2, max(0, distance))


def global_to_local(
    position_xy: Iterable[int],
    ego_xy: Iterable[int],
    map_size: int = 15,
) -> tuple[int, int, bool]:
    px, py = (int(v) for v in position_xy)
    ex, ey = (int(v) for v in ego_xy)
    radius = map_size // 2
    lx, ly = px - ex + radius, py - ey + radius
    valid = 0 <= lx < map_size and 0 <= ly < map_size
    return lx, ly, valid


def action_from_displacement(displacement_xy: Iterable[int]) -> int:
    dx, dy = (int(v) for v in displacement_xy)
    for action, move in enumerate(MOVES):
        if dx == int(move[0]) and dy == int(move[1]):
            return action
    return WAIT


def displacement_from_action(action: int) -> np.ndarray:
    action = int(action)
    if not 0 <= action < len(MOVES):
        raise ValueError(f"Invalid action {action}")
    return MOVES[action].copy()


def pack_agent_payload(
    x: int,
    y: int,
    action_mask: Iterable[int],
    distance_bucket: int,
) -> np.uint32:
    """Packs the 18-bit physical payload into a uint32 storage word.

    Layout, least significant first: x(4), y(4), mask(4), distance(6).
    """
    x, y, distance_bucket = int(x), int(y), int(distance_bucket)
    mask_bits = 0
    for bit, enabled in enumerate(action_mask):
        if int(enabled):
            mask_bits |= 1 << bit
    if not (0 <= x <= 15 and 0 <= y <= 15):
        raise ValueError("x and y must fit in four bits")
    if not 0 <= distance_bucket <= 63:
        raise ValueError("distance_bucket must fit in six bits")
    return np.uint32(x | (y << 4) | (mask_bits << 8) | (distance_bucket << 12))


def unpack_agent_payload(payload: int) -> tuple[int, int, np.ndarray, int]:
    payload = int(payload)
    x = payload & 0xF
    y = (payload >> 4) & 0xF
    mask_bits = (payload >> 8) & 0xF
    distance = (payload >> 12) & 0x3F
    action_mask = np.asarray([(mask_bits >> bit) & 1 for bit in range(4)], dtype=np.uint8)
    return x, y, action_mask, distance
