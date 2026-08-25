from __future__ import annotations

from enum import IntEnum


ACTION_NAMES = ("WAIT", "UP", "DOWN", "LEFT", "RIGHT")
ACTION_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class Action(IntEnum):
    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    PAD = 5


class Outcome(IntEnum):
    SUCCESS = 0
    WAIT = 1
    BLOCKED_OR_OVERRIDDEN = 2
    AT_GOAL = 3
    CONFLICT_RESOLVED = 4
    UNKNOWN = 5
    PAD = 6


class DeltaCTG(IntEnum):
    DECREASE = 0
    SAME = 1
    INCREASE = 2
    BLOCKED = 3
    UNREACHABLE = 4
    PAD = 5


class ConflictType(IntEnum):
    NONE = 0
    VERTEX = 1
    EDGE_SWAP = 2
    CORRIDOR = 3


REASON_NAMES = (
    "PROGRESS_TO_GOAL",
    "WAIT_AT_GOAL",
    "STATIC_BLOCKED",
    "VERTEX_CONFLICT_AVOIDANCE",
    "EDGE_SWAP_AVOIDANCE",
    "YIELD_TO_AGENT",
    "CORRIDOR_PRIORITY",
    "CONGESTION_DETOUR",
    "GOAL_UNBLOCKING",
    "DEADLOCK_RECOVERY",
    "OSCILLATION_RECOVERY",
    "NO_SAFE_ACTION",
)
