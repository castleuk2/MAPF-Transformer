from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import EpisodeData, save_episode, write_manifest
from .geometry import MASK_ACTION_IDS, MOVES, WAIT, bfs_distance_map


def _distinct_cells(rng: np.random.Generator, height: int, width: int, count: int) -> np.ndarray:
    choices = rng.choice(height * width, size=count, replace=False)
    return np.stack([choices // width, choices % width], axis=1).astype(np.int16)


def generate_synthetic_episode(
    seed: int = 0,
    height: int = 17,
    width: int = 17,
    num_agents: int = 4,
    max_steps: int = 48,
) -> EpisodeData:
    """Creates a small collision-free, sequential-priority demonstration.

    This generator is intended for smoke tests. The companion POGEMA project
    contains the production dataset generator and expert-planner adapters.
    """
    rng = np.random.default_rng(seed)
    obstacles = np.zeros((height, width), dtype=np.uint8)
    cells = _distinct_cells(rng, height, width, num_agents * 2)
    positions = cells[:num_agents].copy()
    goals = cells[num_agents:].copy()
    distance_maps = [bfs_distance_map(obstacles, goal) for goal in goals]

    position_history = [positions.copy()]
    action_history: list[np.ndarray] = []
    for step in range(max_steps):
        actions = np.full(num_agents, WAIT, dtype=np.uint8)
        next_positions = positions.copy()
        active = step % num_agents
        if not np.array_equal(positions[active], goals[active]):
            x, y = positions[active]
            current_distance = int(distance_maps[active][x, y])
            for action in MASK_ACTION_IDS:
                dx, dy = MOVES[action]
                candidate = positions[active] + np.asarray([dx, dy], dtype=np.int16)
                cx, cy = int(candidate[0]), int(candidate[1])
                if (
                    0 <= cx < height
                    and 0 <= cy < width
                    and int(distance_maps[active][cx, cy]) == current_distance - 1
                    and not np.any(np.all(positions[np.arange(num_agents) != active] == candidate, axis=1))
                ):
                    actions[active] = action
                    next_positions[active] = candidate
                    break
        positions = next_positions
        action_history.append(actions)
        position_history.append(positions.copy())
        if np.all(positions == goals):
            break

    return EpisodeData(
        obstacles=obstacles,
        positions=np.asarray(position_history, dtype=np.int16),
        goals=goals,
        actions=np.asarray(action_history, dtype=np.uint8),
        metadata={"generator": "mapf_transformer.synthetic", "seed": seed},
    )


def create_synthetic_dataset(
    output_dir: str | Path,
    episodes: int = 4,
    seed: int = 0,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for episode_index in range(episodes):
        episode = generate_synthetic_episode(seed=seed + episode_index)
        path = output_dir / f"episode_{episode_index:05d}.npz"
        save_episode(path, episode)
        paths.append(path)
    manifest = output_dir / "manifest.jsonl"
    write_manifest(paths, manifest)
    return manifest
