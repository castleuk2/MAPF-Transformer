from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    PAD = 5
    UNKNOWN = 6


ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


class DeltaCTG(IntEnum):
    DECREASE = 0
    SAME = 1
    INCREASE = 2
    BLOCKED = 3
    UNREACHABLE = 4
    PAD = 5


class TokenField(IntEnum):
    MAP = 0
    CURRENT_POSITION = 1
    CURRENT_GOAL = 2
    CURRENT_HOPS = 3
    CANDIDATE_WAIT = 4
    CANDIDATE_UP = 5
    CANDIDATE_DOWN = 6
    CANDIDATE_LEFT = 7
    CANDIDATE_RIGHT = 8
    HISTORY_POSITION = 9
    HISTORY_GOAL = 10
    HISTORY_HOPS = 11
    HISTORY_ACTION_OUTCOME = 12
    MESSAGE_QUERY = 13
    NEIGHBOR_MESSAGE = 14
    PAD = 15
    COUNT = 16


ACTION_FIELD_IDS: tuple[int, ...] = (
    int(TokenField.CANDIDATE_WAIT),
    int(TokenField.CANDIDATE_UP),
    int(TokenField.CANDIDATE_DOWN),
    int(TokenField.CANDIDATE_LEFT),
    int(TokenField.CANDIDATE_RIGHT),
)
