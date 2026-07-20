from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .compat import all_done, extract_global_fields, make_pogema_env, normalize_reset, normalize_step
from .config import DatasetGenerationConfig, load_dataset_config, save_config
from .episode_io import build_episode_data, save_episode, write_manifest
from .expert import (
    ExternalCommandPlanner,
    MAPFLNS2Planner,
    PlanningFailure,
    PrioritizedTimeExpandedPlanner,
)


def _build_planner(config: DatasetGenerationConfig, episode_seed: int):
    if config.expert == "external":
        assert config.external_planner_command is not None
        return ExternalCommandPlanner(config.external_planner_command)
    if config.expert == "mapf_lns2":
        return MAPFLNS2Planner(
            binary=config.mapf_lns2_binary,
            cutoff_time=config.mapf_lns2_cutoff_time,
            init_algo=config.mapf_lns2_init_algo,
            replan_algo=config.mapf_lns2_replan_algo,
            destroy_strategy=config.mapf_lns2_destroy_strategy,
            neighbor_size=config.mapf_lns2_neighbor_size,
            max_iterations=config.mapf_lns2_max_iterations,
            screen=config.mapf_lns2_screen,
            seed=episode_seed,
        )
    return PrioritizedTimeExpandedPlanner(
        horizon=config.planner_horizon,
        retries=config.planner_retries,
        seed=episode_seed,
    )


def _execute_plan(env, initial_observations, actions: np.ndarray):
    obstacles, starts, goals = extract_global_fields(initial_observations)
    positions = [starts.copy()]
    executed_actions: list[np.ndarray] = []
    observations = initial_observations
    for action_row in actions:
        observations, rewards, terminated, truncated, infos = normalize_step(
            env.step(np.asarray(action_row, dtype=np.int64).tolist())
        )
        del rewards, infos
        _, current_positions, current_goals = extract_global_fields(observations)
        if not np.array_equal(current_goals, goals):
            raise PlanningFailure("Static MAPF generator observed changing goals")
        positions.append(current_positions.copy())
        executed_actions.append(np.asarray(action_row, dtype=np.uint8).copy())
        if all_done(terminated, truncated):
            break
    action_array = np.asarray(executed_actions, dtype=np.uint8)
    position_array = np.asarray(positions, dtype=np.int16)
    if action_array.shape[0] + 1 != position_array.shape[0]:
        raise RuntimeError("Recorded position/action lengths are inconsistent")
    return obstacles, position_array, goals, action_array


def generate_dataset(config: DatasetGenerationConfig) -> Path:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "generation_config.yaml")
    episode_paths: list[Path] = []
    failures: list[dict[str, str | int]] = []

    for episode_index in range(config.episodes):
        generated = False
        for scenario_attempt in range(config.max_scenario_attempts):
            episode_seed = config.seed + episode_index * config.max_scenario_attempts + scenario_attempt
            env = make_pogema_env(
                num_agents=config.num_agents,
                map_size=config.map_size,
                density=config.density,
                seed=episode_seed,
                max_episode_steps=config.max_episode_steps,
                observation_type=config.observation_type,
                obs_radius=config.obs_radius,
                on_target=config.on_target,
                collision_system=config.collision_system,
            )
            try:
                observations, _ = normalize_reset(env.reset())
                obstacles, starts, goals = extract_global_fields(observations)
                planner = _build_planner(config, episode_seed)
                plan = planner.plan(obstacles, starts, goals)
                if plan.actions.shape[0] > config.max_episode_steps:
                    raise PlanningFailure(
                        f"Expert plan length {plan.actions.shape[0]} exceeds environment limit"
                    )
                obstacles, positions, goals, actions = _execute_plan(env, observations, plan.actions)
                if actions.size == 0 or not np.all(positions[-1] == goals):
                    raise PlanningFailure("Executed expert plan did not reach every goal")
                # Validate that POGEMA execution agrees with the planned collision-free transitions.
                predicted = PrioritizedTimeExpandedPlanner.validate_plan(starts, actions, obstacles)
                if not np.array_equal(predicted, positions):
                    raise PlanningFailure("POGEMA execution differs from the expert plan")

                episode = build_episode_data(
                    obstacles=obstacles,
                    positions=positions,
                    goals=goals,
                    actions=actions,
                    metadata={
                        "episode_index": episode_index,
                        "seed": episode_seed,
                        "planner": plan.planner,
                        "priority_order": plan.priority_order,
                        "pogema_action_order": ["WAIT", "UP", "DOWN", "LEFT", "RIGHT"],
                    },
                    precompute_tracking=config.precompute_tracking,
                    tracking_grace_steps=config.tracking_grace_steps,
                )
                path = output_dir / f"episode_{episode_index:07d}.npz"
                save_episode(path, episode, compress=config.compress)
                episode_paths.append(path)
                generated = True
                print(
                    f"episode={episode_index + 1}/{config.episodes} seed={episode_seed} "
                    f"steps={actions.shape[0]} planner={plan.planner}",
                    flush=True,
                )
                break
            except (PlanningFailure, RuntimeError, ValueError, KeyError) as error:
                failures.append(
                    {
                        "episode_index": episode_index,
                        "seed": episode_seed,
                        "error": str(error),
                    }
                )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        if not generated:
            failure_path = output_dir / "failures.json"
            failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
            raise RuntimeError(
                f"Could not generate episode {episode_index} after "
                f"{config.max_scenario_attempts} scenarios. See {failure_path}."
            )

    manifest = output_dir / "manifest.jsonl"
    write_manifest(episode_paths, manifest)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Generated {len(episode_paths)} episodes: {manifest}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MAPF Transformer training trajectories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--expert", choices=["mapf_lns2", "prioritized", "external"], default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_dataset_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.seed is not None:
        config.seed = args.seed
    if args.expert:
        config.expert = args.expert
    generate_dataset(config)


if __name__ == "__main__":
    main()
