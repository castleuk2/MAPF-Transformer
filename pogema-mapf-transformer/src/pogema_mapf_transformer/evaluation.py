from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .compat import all_done, extract_global_fields, make_pogema_env
from .config import EvaluationConfig, load_evaluation_config
from .env_adapter import POGEMAMAPFTransformerAdapter
from .policy_adapter import MAPFTransformerPOGEMAPolicy


def evaluate(config: EvaluationConfig) -> dict[str, float | int]:
    config.validate()
    policy = MAPFTransformerPOGEMAPolicy(
        checkpoint_path=config.checkpoint,
        device=config.device,
        sample_actions=config.sample_actions,
        temperature=config.temperature,
        seed=config.seed,
    )
    successes = 0
    total_steps = 0
    failed_moves = 0
    total_commands = 0

    svg_dir = Path(config.save_svg_dir) if config.save_svg_dir else None
    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in range(config.episodes):
        pogema_env = make_pogema_env(
                num_agents=config.num_agents,
                map_size=config.map_size,
                density=config.density,
                seed=config.seed + episode_index,
                max_episode_steps=config.max_episode_steps,
                observation_type="MAPF",
                obs_radius=7,
                on_target="nothing",
                collision_system=config.collision_system,
                render_mode=config.render_mode,
            )
        if svg_dir is not None:
            # POGEMA records agent states only after animation is enabled and reset.
            pogema_env.enable_animation()
        env = POGEMAMAPFTransformerAdapter(pogema_env)
        policy.reset()
        observations, _ = env.reset()
        episode_steps = 0
        terminated = truncated = np.zeros(config.num_agents, dtype=bool)
        for _ in range(config.max_episode_steps):
            actions = policy.act(observations)
            observations, rewards, terminated, truncated, info, transition = env.step(actions)
            del rewards, info
            total_commands += len(actions)
            failed_moves += int(
                np.sum((transition.commanded_actions != 0) & (transition.actual_moves == 0))
            )
            episode_steps += 1
            if config.render_mode:
                rendered = env.render()
                if rendered is not None:
                    print(rendered)
            if all_done(terminated, truncated):
                break
        _, positions, goals = extract_global_fields(observations)
        success = bool(np.all(positions == goals))
        successes += int(success)
        total_steps += episode_steps
        print(
            f"episode={episode_index + 1}/{config.episodes} success={success} steps={episode_steps}",
            flush=True,
        )
        if svg_dir is not None:
            svg_path = svg_dir / f"episode_{episode_index:05d}_seed_{config.seed + episode_index}.svg"
            env.save_animation(str(svg_path))
            print(f"svg={svg_path}", flush=True)
        env.close()

    result: dict[str, float | int] = {
        "episodes": config.episodes,
        "successes": successes,
        "success_rate": successes / config.episodes,
        "average_steps": total_steps / config.episodes,
        "failed_move_rate": failed_moves / max(1, total_commands),
    }
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate MAPF Transformer in POGEMA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--render-mode", choices=["ansi"], default=None)
    parser.add_argument("--save-svg-dir", default=None)
    args = parser.parse_args(argv)
    config = load_evaluation_config(args.config)
    if args.checkpoint:
        config.checkpoint = args.checkpoint
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.device:
        config.device = args.device
    if args.render_mode:
        config.render_mode = args.render_mode
    if args.save_svg_dir:
        config.save_svg_dir = args.save_svg_dir
    result = evaluate(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
