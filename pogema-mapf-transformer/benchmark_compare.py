from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from pogema_mapf_transformer.compat import extract_global_fields, make_pogema_env, normalize_reset, normalize_step
from pogema_mapf_transformer.expert import MAPFLNS2Planner, PlanningFailure
from pogema_mapf_transformer.policy_adapter import MAPFTransformerPOGEMAPolicy


SUITES = {
    "random": "01-random",
    "mazes": "02-mazes",
    "warehouse": "03-warehouse",
    "movingai": "04-movingai",
    "puzzles": "05-puzzles",
}
RESULT_FIELDS = [
    "model", "suite", "map_name", "seed", "num_agents", "max_episode_steps",
    "CSR", "ISR", "SoC", "makespan", "runtime", "episode_steps", "status",
]


@dataclass(frozen=True)
class Scenario:
    suite: str
    map_name: str
    map_data: str
    seed: int
    num_agents: int
    max_episode_steps: int
    collision_system: str

    def key(self, model: str) -> tuple[str, str, str, int, int]:
        return model, self.suite, self.map_name, self.seed, self.num_agents


def grid_values(value: Any) -> list[Any]:
    if isinstance(value, dict) and "grid_search" in value:
        return list(value["grid_search"])
    return [value]


def load_scenarios(config_root: Path, suite_names: Iterable[str]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for suite in suite_names:
        folder = config_root / SUITES[suite]
        with (folder / "maps.yaml").open(encoding="utf-8") as stream:
            maps = yaml.safe_load(stream)
        with (folder / f"{folder.name}.yaml").open(encoding="utf-8") as stream:
            environment = yaml.safe_load(stream)["environment"]
        for map_name, seed, num_agents in itertools.product(
            grid_values(environment["map_name"]),
            grid_values(environment["seed"]),
            grid_values(environment["num_agents"]),
        ):
            scenarios.append(
                Scenario(
                    suite=suite,
                    map_name=str(map_name),
                    map_data=str(maps[str(map_name)]),
                    seed=int(seed),
                    num_agents=int(num_agents),
                    max_episode_steps=int(environment["max_episode_steps"]),
                    collision_system=str(environment["collision_system"]),
                )
            )
    return scenarios


class CurrentPolicy:
    def __init__(self, checkpoint: str, device: str) -> None:
        self.impl = MAPFTransformerPOGEMAPolicy(checkpoint, device=device, sample_actions=False)

    def reset(self) -> None:
        self.impl.reset()

    def act(self, observations: Any) -> list[int]:
        return self.impl.act(observations)


class MAPFGPTPolicy:
    def __init__(self, checkpoint: str, device: str) -> None:
        try:
            from mapf_gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
        except ImportError as error:
            raise RuntimeError(
                "MAPF-GPT is not installed. Run: MAPF/bin/python -m pip install "
                "'git+https://github.com/CognitiveAISystems/MAPF-GPT.git'"
            ) from error
        # Match the official benchmark: build/import the C++ observation extension
        # before RuntimeMetricWrapper-style policy timing begins.
        MAPFGPTInference.build()
        self.impl = MAPFGPTInference(
            MAPFGPTInferenceConfig(path_to_weights=checkpoint, device=device)
        )

    def reset(self) -> None:
        self.impl.reset_states()

    def act(self, observations: Any) -> list[int]:
        return self.impl.act(observations)


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_scenario(policy: Any, scenario: Scenario, device: str) -> dict[str, float | int]:
    rows = scenario.map_data.splitlines()
    env = make_pogema_env(
        num_agents=scenario.num_agents,
        map_size=max(len(rows), max(map(len, rows))),
        density=0.0,
        seed=scenario.seed,
        max_episode_steps=scenario.max_episode_steps,
        observation_type="MAPF",
        obs_radius=7,
        on_target="nothing",
        collision_system=scenario.collision_system,
        map_data=scenario.map_data,
        map_name=scenario.map_name,
    )
    policy.reset()
    observations, _ = normalize_reset(env.reset())
    arrival_steps: list[int | None] = [None] * scenario.num_agents
    runtime = 0.0
    episode_steps = 0
    try:
        for step in range(1, scenario.max_episode_steps + 1):
            synchronize(device)
            started = time.perf_counter()
            actions = policy.act(observations)
            synchronize(device)
            runtime += time.perf_counter() - started
            observations, _, terminated, truncated, _ = normalize_step(env.step(actions))
            episode_steps = step
            _, positions, goals = extract_global_fields(observations)
            on_goal = np.all(positions == goals, axis=1)
            for index, reached in enumerate(on_goal):
                if reached and arrival_steps[index] is None:
                    arrival_steps[index] = step
                elif not reached:
                    arrival_steps[index] = None
            if bool(np.all(np.asarray(terminated) | np.asarray(truncated))):
                break
        _, positions, goals = extract_global_fields(observations)
        on_goal = np.all(positions == goals, axis=1)
        final_arrivals = [
            int(arrival) if arrival is not None else episode_steps
            for arrival in arrival_steps
        ]
        return {
            "CSR": float(np.all(on_goal)),
            "ISR": float(np.mean(on_goal)),
            "SoC": int(sum(final_arrivals)),
            "makespan": int(max(final_arrivals)),
            "runtime": runtime,
            "episode_steps": episode_steps,
            "status": "completed",
        }
    finally:
        env.close()


def run_mapf_lns2_scenario(
    scenario: Scenario,
    binary: str,
    cutoff_time: float,
    max_iterations: int,
) -> dict[str, float | int | str]:
    """Plan once with MAPF-LNS2 and validate its actions in the same POGEMA env."""
    rows = scenario.map_data.splitlines()
    env = make_pogema_env(
        num_agents=scenario.num_agents,
        map_size=max(len(rows), max(map(len, rows))),
        density=0.0,
        seed=scenario.seed,
        max_episode_steps=scenario.max_episode_steps,
        observation_type="MAPF",
        obs_radius=7,
        on_target="nothing",
        collision_system=scenario.collision_system,
        map_data=scenario.map_data,
        map_name=scenario.map_name,
    )
    observations, _ = normalize_reset(env.reset())
    obstacles, starts, goals = extract_global_fields(observations)
    planner = MAPFLNS2Planner(
        binary=binary,
        cutoff_time=cutoff_time,
        init_algo="PP",
        replan_algo="PP",
        destroy_strategy="Adaptive",
        neighbor_size=8,
        max_iterations=max_iterations,
        screen=0,
        seed=scenario.seed,
    )
    started = time.perf_counter()
    try:
        plan = planner.plan(obstacles, starts, goals)
        status = "completed"
    except PlanningFailure as error:
        runtime = time.perf_counter() - started
        env.close()
        return {
            "CSR": 0.0,
            "ISR": 0.0,
            "SoC": scenario.max_episode_steps * scenario.num_agents,
            "makespan": scenario.max_episode_steps,
            "runtime": runtime,
            "episode_steps": 0,
            "status": f"planner_failed:{type(error).__name__}",
        }
    runtime = time.perf_counter() - started

    arrival_steps: list[int | None] = [None] * scenario.num_agents
    episode_steps = 0
    try:
        for step in range(1, scenario.max_episode_steps + 1):
            actions = (
                plan.actions[step - 1].tolist()
                if step <= plan.actions.shape[0]
                else [0] * scenario.num_agents
            )
            observations, _, terminated, truncated, _ = normalize_step(env.step(actions))
            episode_steps = step
            _, positions, current_goals = extract_global_fields(observations)
            on_goal = np.all(positions == current_goals, axis=1)
            for index, reached in enumerate(on_goal):
                if reached and arrival_steps[index] is None:
                    arrival_steps[index] = step
                elif not reached:
                    arrival_steps[index] = None
            if bool(np.all(np.asarray(terminated) | np.asarray(truncated))):
                break
        _, positions, current_goals = extract_global_fields(observations)
        on_goal = np.all(positions == current_goals, axis=1)
        final_arrivals = [
            int(arrival) if arrival is not None else episode_steps
            for arrival in arrival_steps
        ]
        return {
            "CSR": float(np.all(on_goal)),
            "ISR": float(np.mean(on_goal)),
            "SoC": int(sum(final_arrivals)),
            "makespan": int(max(final_arrivals)),
            "runtime": runtime,
            "episode_steps": episode_steps,
            "status": status,
        }
    finally:
        env.close()


def read_completed(path: Path) -> tuple[list[dict[str, str]], set[tuple[str, str, str, int, int]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = {
        (row["model"], row["suite"], row["map_name"], int(row["seed"]), int(row["num_agents"]))
        for row in rows
    }
    return rows, keys


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["suite"]), int(row["num_agents"]))].append(row)
    summary = []
    for (model, suite, num_agents), items in sorted(groups.items()):
        summary.append(
            {
                "model": model,
                "suite": suite,
                "num_agents": num_agents,
                "episodes": len(items),
                "CSR": float(np.mean([float(item["CSR"]) for item in items])),
                "ISR": float(np.mean([float(item["ISR"]) for item in items])),
                "SoC": float(np.mean([float(item["SoC"]) for item in items])),
                "makespan": float(np.mean([float(item["makespan"]) for item in items])),
                "runtime": float(np.mean([float(item["runtime"]) for item in items])),
            }
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MAPF policies and MAPF-LNS2 on official eval configs")
    parser.add_argument("--config-root", type=Path, default=Path("configs/mapf_gpt_eval"))
    parser.add_argument("--suites", nargs="+", choices=list(SUITES), default=list(SUITES))
    parser.add_argument("--models", nargs="+", choices=["mapf_transformer", "mapf_gpt_6m", "mapf_lns2"], default=["mapf_transformer", "mapf_gpt_6m"])
    parser.add_argument("--current-checkpoint", default="../mapf-transformer-policy/runs/mapf_lns2_1h/best.pt")
    parser.add_argument("--mapf-gpt-checkpoint", default="weights/MAPF-GPT-6M.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mapf_gpt_comparison"))
    parser.add_argument("--agent-counts", nargs="+", type=int, default=None)
    parser.add_argument("--map-limit", type=int, default=None, help="Keep the first N maps per suite")
    parser.add_argument("--seed-limit", type=int, default=None, help="Keep the first N seeds per suite")
    parser.add_argument("--mapf-lns2-binary", default="external/MAPF-LNS2/lns")
    parser.add_argument("--mapf-lns2-cutoff", type=float, default=10.0)
    parser.add_argument("--mapf-lns2-max-iterations", type=int, default=1_000_000_000)
    parser.add_argument(
        "--mapf-lns2-workers",
        type=int,
        default=1,
        help="Independent MAPF-LNS2 scenarios to solve concurrently",
    )
    args = parser.parse_args()
    if args.mapf_lns2_workers <= 0:
        parser.error("--mapf-lns2-workers must be positive")

    scenarios = load_scenarios(args.config_root, args.suites)
    if args.agent_counts:
        scenarios = [scenario for scenario in scenarios if scenario.num_agents in args.agent_counts]
    if args.map_limit is not None or args.seed_limit is not None:
        kept: list[Scenario] = []
        seen_maps: dict[str, list[str]] = defaultdict(list)
        seen_seeds: dict[str, list[int]] = defaultdict(list)
        for scenario in scenarios:
            if scenario.map_name not in seen_maps[scenario.suite]:
                seen_maps[scenario.suite].append(scenario.map_name)
            if scenario.seed not in seen_seeds[scenario.suite]:
                seen_seeds[scenario.suite].append(scenario.seed)
            if args.map_limit is not None and scenario.map_name not in seen_maps[scenario.suite][:args.map_limit]:
                continue
            if args.seed_limit is not None and scenario.seed not in seen_seeds[scenario.suite][:args.seed_limit]:
                continue
            kept.append(scenario)
        scenarios = kept

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.csv"
    old_rows, completed = read_completed(episodes_path)
    mode = "a" if episodes_path.exists() else "w"
    with episodes_path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if mode == "w":
            writer.writeheader()
        for model_name in args.models:
            if model_name == "mapf_transformer":
                policy = CurrentPolicy(args.current_checkpoint, args.device)
            elif model_name == "mapf_gpt_6m":
                policy = MAPFGPTPolicy(args.mapf_gpt_checkpoint, args.device)
            else:
                policy = None
            pending = [scenario for scenario in scenarios if scenario.key(model_name) not in completed]
            print(f"model={model_name} scenarios={len(pending)} device={args.device}", flush=True)

            def record(index: int, scenario: Scenario, metrics: dict[str, Any]) -> None:
                row = {
                    "model": model_name,
                    "suite": scenario.suite,
                    "map_name": scenario.map_name,
                    "seed": scenario.seed,
                    "num_agents": scenario.num_agents,
                    "max_episode_steps": scenario.max_episode_steps,
                    **metrics,
                }
                writer.writerow(row)
                stream.flush()
                old_rows.append({key: str(value) for key, value in row.items()})
                print(
                    f"[{index}/{len(pending)}] {scenario.suite} {scenario.map_name} "
                    f"seed={scenario.seed} agents={scenario.num_agents} "
                    f"CSR={metrics['CSR']:.0f} ISR={metrics['ISR']:.3f} "
                    f"SoC={metrics['SoC']} makespan={metrics['makespan']} "
                    f"runtime={metrics['runtime']:.4f}s",
                    flush=True,
                )

            if model_name == "mapf_lns2" and args.mapf_lns2_workers > 1:
                print(
                    f"mapf_lns2_workers={args.mapf_lns2_workers} "
                    f"cutoff={args.mapf_lns2_cutoff}s",
                    flush=True,
                )
                with ProcessPoolExecutor(max_workers=args.mapf_lns2_workers) as executor:
                    futures = {
                        executor.submit(
                            run_mapf_lns2_scenario,
                            scenario,
                            args.mapf_lns2_binary,
                            args.mapf_lns2_cutoff,
                            args.mapf_lns2_max_iterations,
                        ): scenario
                        for scenario in pending
                    }
                    for index, future in enumerate(as_completed(futures), 1):
                        scenario = futures[future]
                        record(index, scenario, future.result())
            else:
                for index, scenario in enumerate(pending, 1):
                    if model_name == "mapf_lns2":
                        metrics = run_mapf_lns2_scenario(
                            scenario,
                            binary=args.mapf_lns2_binary,
                            cutoff_time=args.mapf_lns2_cutoff,
                            max_iterations=args.mapf_lns2_max_iterations,
                        )
                    else:
                        metrics = run_scenario(policy, scenario, args.device)
                    record(index, scenario, metrics)
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_summary(old_rows, args.output_dir)
    print(f"episodes={episodes_path}")
    print(f"summary={args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
