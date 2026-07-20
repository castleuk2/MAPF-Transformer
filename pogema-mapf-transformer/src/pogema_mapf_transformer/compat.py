from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def require_pogema() -> tuple[Any, Any]:
    try:
        from pogema import GridConfig, pogema_v0
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "POGEMA is not installed. Install this project with the 'pogema' extra: "
            "python -m pip install -e '.[pogema]'"
        ) from error
    return GridConfig, pogema_v0


def make_pogema_env(
    *,
    num_agents: int,
    map_size: int,
    density: float,
    seed: int,
    max_episode_steps: int,
    observation_type: str = "MAPF",
    obs_radius: int = 7,
    on_target: str = "nothing",
    collision_system: str = "soft",
    render_mode: str | None = None,
    map_data: str | list[list[int]] | None = None,
    map_name: str | None = None,
) -> Any:
    GridConfig, pogema_v0 = require_pogema()
    kwargs = dict(
        num_agents=num_agents,
        size=map_size,
        density=density,
        seed=seed,
        max_episode_steps=max_episode_steps,
        observation_type=observation_type,
        obs_radius=obs_radius,
        on_target=on_target,
        collision_system=collision_system,
    )
    if map_data is not None:
        kwargs["map"] = map_data
        kwargs["map_name"] = map_name
    try:
        grid_config = GridConfig(**kwargs)
    except TypeError:
        # Compatibility with versions where radius or episode limit are wrapper arguments.
        reduced = dict(kwargs)
        reduced.pop("obs_radius", None)
        grid_config = GridConfig(**reduced)
    try:
        return pogema_v0(grid_config=grid_config, render_mode=render_mode)
    except TypeError:
        try:
            return pogema_v0(grid_config=grid_config)
        except TypeError:
            return pogema_v0(grid_config)


def normalize_reset(result: Any) -> tuple[Any, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def normalize_step(result: Any) -> tuple[Any, Any, Any, Any, Any]:
    if not isinstance(result, tuple):
        raise TypeError("POGEMA step must return a tuple")
    if len(result) == 5:
        return result
    if len(result) == 4:
        observations, rewards, dones, infos = result
        truncated = np.zeros_like(np.asarray(dones), dtype=bool)
        return observations, rewards, dones, truncated, infos
    raise ValueError(f"Unsupported POGEMA step return length: {len(result)}")


def extract_global_fields(
    observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation_list = [observations] if isinstance(observations, Mapping) else list(observations)
    if not observation_list:
        raise ValueError("observations must not be empty")
    first = observation_list[0]
    required = ("global_obstacles", "global_xy", "global_target_xy")
    missing = [key for key in required if key not in first]
    if missing:
        raise KeyError(
            "POGEMA MAPF observations are required; missing fields: " + ", ".join(missing)
        )
    first_position = np.asarray(first["global_xy"], dtype=np.int16)
    first_goal = np.asarray(first["global_target_xy"], dtype=np.int16)
    positions = (
        np.stack([np.asarray(obs["global_xy"], dtype=np.int16) for obs in observation_list])
        if first_position.ndim == 1
        else first_position
    )
    goals = (
        np.stack([np.asarray(obs["global_target_xy"], dtype=np.int16) for obs in observation_list])
        if first_goal.ndim == 1
        else first_goal
    )
    return np.asarray(first["global_obstacles"], dtype=np.uint8), positions, goals


def all_done(terminated: Any, truncated: Any) -> bool:
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    return bool(np.all(term | trunc))
