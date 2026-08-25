from __future__ import annotations

from enum import IntEnum
from typing import Final


class Action(IntEnum):
    """POGEMA-compatible cardinal action order."""

    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


# Deltas are expressed in array coordinates: row grows downward, column rightward.
ACTION_TO_DELTA: Final[dict[Action, tuple[int, int]]] = {
    Action.WAIT: (0, 0),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


def coerce_action(action: Action | int | str) -> Action:
    if isinstance(action, Action):
        return action
    if isinstance(action, str):
        normalized = action.strip().upper()
        try:
            return Action[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown action string: {action!r}") from exc
    try:
        return Action(int(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown action value: {action!r}") from exc
