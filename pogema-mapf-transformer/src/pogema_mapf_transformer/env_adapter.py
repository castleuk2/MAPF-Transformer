from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mapf_transformer.features import TransitionOutcome
from mapf_transformer.geometry import WAIT, action_from_displacement

from .compat import extract_global_fields, normalize_reset, normalize_step


@dataclass(slots=True)
class TransitionBatch:
    commanded_actions: np.ndarray
    actual_moves: np.ndarray
    displacements: np.ndarray
    outcomes: np.ndarray
    previous_positions: np.ndarray
    current_positions: np.ndarray


class POGEMAMAPFTransformerAdapter:
    """Adds explicit execution feedback and global MAPF-state validation to POGEMA."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.previous_positions: np.ndarray | None = None
        self.last_observations: Any = None

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        observations, info = normalize_reset(self.env.reset(*args, **kwargs))
        _, positions, _ = extract_global_fields(observations)
        self.previous_positions = positions.copy()
        self.last_observations = observations
        return observations, info

    def step(self, actions: Sequence[int]) -> tuple[Any, Any, Any, Any, Any, TransitionBatch]:
        actions_array = np.asarray(actions, dtype=np.int64)
        if self.previous_positions is None:
            raise RuntimeError("reset() must be called before step()")
        observations, rewards, terminated, truncated, info = normalize_step(
            self.env.step(actions_array.tolist())
        )
        _, current_positions, _ = extract_global_fields(observations)
        if current_positions.shape[0] != actions_array.shape[0]:
            raise ValueError("The number of actions does not match the agent count")
        displacements = current_positions - self.previous_positions
        actual_moves = np.asarray(
            [action_from_displacement(delta) for delta in displacements], dtype=np.int64
        )
        outcomes = np.empty(actions_array.shape[0], dtype=np.int64)
        for index, commanded in enumerate(actions_array):
            if int(commanded) == WAIT:
                outcomes[index] = int(TransitionOutcome.WAIT)
            elif int(actual_moves[index]) == int(commanded):
                outcomes[index] = int(TransitionOutcome.SUCCESS)
            else:
                outcomes[index] = int(TransitionOutcome.FAILED)
        transition = TransitionBatch(
            commanded_actions=actions_array,
            actual_moves=actual_moves,
            displacements=displacements.astype(np.int16),
            outcomes=outcomes,
            previous_positions=self.previous_positions.copy(),
            current_positions=current_positions.copy(),
        )
        self.previous_positions = current_positions.copy()
        self.last_observations = observations
        return observations, rewards, terminated, truncated, info, transition

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def render(self, *args: Any, **kwargs: Any) -> Any:
        render = getattr(self.env, "render", None)
        if not callable(render):
            return None
        return render(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
