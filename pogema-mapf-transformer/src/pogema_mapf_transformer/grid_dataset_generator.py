from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
from itertools import islice, product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .compat import extract_global_fields, make_pogema_env, normalize_reset
from .config import DatasetGenerationConfig
from .dataset_generator import _build_planner, _execute_plan
from .episode_io import build_episode_data, save_episode
from .expert import PlanningFailure, PrioritizedTimeExpandedPlanner


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return values


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _load_maps(path: Path) -> dict[str, str | list[list[int]]]:
    maps = dict(_load_yaml(path))
    if not maps:
        raise ValueError(f"Map catalog is empty: {path}")
    return maps


def _planner_config(environment: Mapping[str, Any], algorithm: Mapping[str, Any]) -> DatasetGenerationConfig:
    name = str(algorithm.get("name", "MAPF-LNS2"))
    if name != "MAPF-LNS2":
        raise ValueError(f"Unsupported dataset algorithm: {name}")
    return DatasetGenerationConfig(
        episodes=1,
        num_agents=1,
        map_size=int(environment.get("map_size", 21)),
        density=float(environment.get("density", 0.25)),
        max_episode_steps=int(environment.get("max_episode_steps", 128)),
        observation_type="MAPF",
        obs_radius=int(environment.get("obs_radius", 7)),
        on_target=str(environment.get("on_target", "nothing")),
        collision_system=str(environment.get("collision_system", "soft")),
        expert="mapf_lns2",
        mapf_lns2_binary=str(algorithm.get("binary", "external/MAPF-LNS2/lns")),
        mapf_lns2_cutoff_time=float(algorithm.get("cutoff_time", 10.0)),
        mapf_lns2_init_algo=str(algorithm.get("init_algo", "PP")),
        mapf_lns2_replan_algo=str(algorithm.get("replan_algo", "PP")),
        mapf_lns2_destroy_strategy=str(algorithm.get("destroy_strategy", "Adaptive")),
        mapf_lns2_neighbor_size=int(algorithm.get("neighbor_size", 8)),
        mapf_lns2_max_iterations=int(algorithm.get("max_iterations", 0)),
        mapf_lns2_screen=int(algorithm.get("screen", 0)),
        precompute_tracking=bool(environment.get("precompute_tracking", True)),
        tracking_grace_steps=int(environment.get("tracking_grace_steps", 1)),
        compress=bool(environment.get("compress", True)),
    )


def _write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _write_results(rows: list[dict[str, Any]], output_root: Path) -> None:
    fields = ["episode_index", "split", "family", "subset", "map_name", "seed", "num_agents",
              "status", "makespan", "soc", "rectangular_actions", "goal_waits_removed",
              "num_samples", "planner", "path", "error"]
    with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, int]:
        generated = [item for item in items if item["status"] == "generated"]
        return {
            "attempted_episodes": len(items),
            "generated_episodes": len(generated),
            "failed_episodes": len(items) - len(generated),
            "rectangular_actions": sum(int(item.get("rectangular_actions", 0)) for item in generated),
            "goal_waits_removed": sum(int(item.get("goal_waits_removed", 0)) for item in generated),
            "training_samples": sum(int(item["num_samples"]) for item in generated),
        }

    def grouped(field: str) -> dict[str, dict[str, int]]:
        keys = sorted({str(row[field]) for row in rows})
        return {key: summarize([row for row in rows if str(row[field]) == key]) for key in keys}

    summary: dict[str, Any] = {
        "sample_definition": "one ego-agent action before that agent's final goal arrival",
        "sample_formula_per_episode": "sum(arrival_steps) = SoC",
        "totals": summarize(rows),
        "subsets": grouped("subset"),
        "families": grouped("family"),
        "splits": grouped("split"),
    }
    train_samples = summary["subsets"].get("train", {}).get("training_samples", 0)
    val_samples = summary["subsets"].get("val", {}).get("training_samples", 0)
    summary["train_val_sample_ratio"] = train_samples / val_samples if val_samples else None
    return summary


def _print_views(rows: list[dict[str, Any]], views: Mapping[str, Any]) -> None:
    for view_name, view in views.items():
        if not isinstance(view, Mapping) or view.get("type", "tabular") != "tabular":
            continue
        group_by = [str(value) for value in view.get("group_by", ["split"])]
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(tuple(row.get(key) for key in group_by), []).append(row)
        print(f"\n[{view_name}] " + " | ".join(
            group_by + ["generated", "failed", "samples", "avg_makespan"]
        ))
        for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
            generated = [item for item in items if item["status"] == "generated"]
            average = sum(int(item["makespan"]) for item in generated) / len(generated) if generated else 0.0
            print(" | ".join([*(str(value) for value in key), str(len(generated)),
                              str(len(items) - len(generated)),
                              str(sum(int(item["num_samples"]) for item in generated)),
                              f"{average:.2f}"]))


def _integer_grid(values: Any, field_name: str) -> list[int]:
    if isinstance(values, Mapping):
        start = int(values.get("start", 0))
        stop = int(values["stop"])
        step = int(values.get("step", 1))
        return list(range(start, stop, step))
    if isinstance(values, list):
        return [int(value) for value in values]
    raise ValueError(f"{field_name} must be a list or start/stop range")


def _select_map_names(catalog: Mapping[str, Any], split: Mapping[str, Any]) -> list[str]:
    """Select catalog maps explicitly, by catalog position, or trailing numeric ID."""
    selectors = [name for name in ("maps", "map_indices", "map_ids") if name in split]
    if len(selectors) > 1:
        raise ValueError(f"Use only one map selector, not {selectors}")
    catalog_names = [str(name) for name in catalog]
    if not selectors:
        return catalog_names
    selector = selectors[0]
    if selector == "maps":
        return [str(name) for name in split[selector]]
    selected_numbers = set(_integer_grid(split[selector], selector))
    if selector == "map_indices":
        invalid = sorted(index for index in selected_numbers if index < 0 or index >= len(catalog_names))
        if invalid:
            raise ValueError(
                f"map_indices outside catalog range 0..{len(catalog_names) - 1}: {invalid}"
            )
        return [name for index, name in enumerate(catalog_names) if index in selected_numbers]
    selected: list[str] = []
    for name in catalog_names:
        match = re.search(r"(\d+)$", name)
        if match and int(match.group(1)) in selected_numbers:
            selected.append(name)
    return selected


def _run_scenario(task: dict[str, Any]) -> dict[str, Any]:
    """Runs one isolated map/seed/agent scenario inside a worker process."""
    episode_index = int(task["episode_index"])
    split_name = str(task["split"])
    family = str(task["family"])
    subset = str(task["subset"])
    map_name = str(task["map_name"])
    seed = int(task["seed"])
    num_agents = int(task["num_agents"])
    planner_config: DatasetGenerationConfig = task["planner_config"]
    split_dir = Path(task["split_dir"])
    output_root = Path(task["output_root"])
    row: dict[str, Any] = {
        "episode_index": episode_index, "split": split_name, "family": family,
        "subset": subset, "map_name": map_name, "seed": seed,
        "num_agents": num_agents, "status": "failed", "makespan": 0,
        "soc": 0, "rectangular_actions": 0, "goal_waits_removed": 0, "num_samples": 0,
        "planner": "mapf_lns2", "path": "", "error": "",
    }
    record = aggregate_record = None
    env = make_pogema_env(
        num_agents=num_agents,
        map_size=planner_config.map_size,
        density=planner_config.density,
        seed=seed,
        max_episode_steps=planner_config.max_episode_steps,
        observation_type=planner_config.observation_type,
        obs_radius=planner_config.obs_radius,
        on_target=planner_config.on_target,
        collision_system=planner_config.collision_system,
        map_data=task["map_data"],
        map_name=map_name,
    )
    try:
        observations, _ = normalize_reset(env.reset())
        obstacles, starts, goals = extract_global_fields(observations)
        planner = _build_planner(planner_config, seed)
        plan = planner.plan(obstacles, starts, goals)
        if plan.actions.shape[0] > planner_config.max_episode_steps:
            raise PlanningFailure("Expert plan exceeds max_episode_steps")
        obstacles, positions, goals, actions = _execute_plan(env, observations, plan.actions)
        if actions.size == 0 or not np.all(positions[-1] == goals):
            raise PlanningFailure("Executed MAPF-LNS2 plan did not reach every goal")
        predicted = PrioritizedTimeExpandedPlanner.validate_plan(starts, actions, obstacles)
        if not np.array_equal(predicted, positions):
            raise PlanningFailure("POGEMA execution differs from MAPF-LNS2 paths")
        metadata = {
            "episode_index": episode_index, "split": split_name,
            "map_family": family, "subset": subset, "map_name": map_name,
            "seed": seed, "num_agents": num_agents, "planner": plan.planner,
            "pogema_action_order": ["WAIT", "UP", "DOWN", "LEFT", "RIGHT"],
        }
        episode = build_episode_data(
            obstacles, positions, goals, actions, metadata=metadata,
            precompute_tracking=planner_config.precompute_tracking,
            tracking_grace_steps=planner_config.tracking_grace_steps,
        )
        filename = f"episode_{episode_index:07d}_{map_name}_s{seed}_n{num_agents}.npz"
        path = split_dir / filename
        arrival_steps = episode.get_arrival_steps()
        makespan = int(actions.shape[0])
        rectangular_actions = makespan * num_agents
        num_samples = int(arrival_steps.sum())
        goal_waits_removed = rectangular_actions - num_samples
        episode.metadata.update({
            "arrival_steps": arrival_steps.tolist(),
            "soc": num_samples,
            "makespan": makespan,
            "goal_waits_removed": goal_waits_removed,
        })
        save_episode(path, episode, compress=planner_config.compress)
        record = {
            "path": filename, "time_steps": makespan, "makespan": makespan,
            "num_agents": num_agents, "arrival_steps": arrival_steps.tolist(),
            "soc": num_samples, "num_samples": num_samples,
            "rectangular_actions": rectangular_actions,
            "goal_waits_removed": goal_waits_removed,
            "map_name": map_name, "seed": seed,
            "split": split_name, "map_family": family, "subset": subset,
            "planner": plan.planner,
        }
        aggregate_record = dict(record)
        aggregate_record["path"] = str(path.relative_to(output_root))
        row.update(
            status="generated", makespan=makespan, soc=num_samples,
            rectangular_actions=rectangular_actions,
            goal_waits_removed=goal_waits_removed, num_samples=num_samples, path=filename,
        )
    except (PlanningFailure, RuntimeError, ValueError, KeyError) as error:
        row["error"] = str(error)
    finally:
        env.close()
    return {"row": row, "record": record, "aggregate_record": aggregate_record}


def generate_grid_dataset(
    config_path: str | Path,
    max_scenarios_per_split: int | None = None,
    num_processes: int | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    values = _load_yaml(config_path)
    base = config_path.parent
    output_root = _resolve(str(values.get("output_root", "../data/mapf_lns2")), base)
    output_root.mkdir(parents=True, exist_ok=True)
    environment = values.get("environment", {})
    algorithms = values.get("algorithms", {})
    if not isinstance(environment, Mapping) or not isinstance(algorithms, Mapping):
        raise ValueError("environment and algorithms must be mappings")
    algorithm_name = str(values.get("algorithm", "MAPF-LNS2"))
    if algorithm_name not in algorithms:
        raise ValueError(f"Algorithm configuration not found: {algorithm_name}")
    algorithm = algorithms[algorithm_name]
    if not isinstance(algorithm, Mapping):
        raise ValueError("Selected algorithm configuration must be a mapping")
    planner_config = _planner_config(environment, algorithm)
    planner_config.mapf_lns2_binary = str(_resolve(planner_config.mapf_lns2_binary, base))
    planner_config.validate()
    generation = values.get("generation", {})
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be a mapping")
    workers = int(num_processes if num_processes is not None else generation.get("num_processes", 24))
    if workers <= 0:
        raise ValueError("num_processes must be positive")
    print(f"Dataset generation workers: {workers}", flush=True)

    splits = values.get("splits", {})
    if not isinstance(splits, Mapping) or not splits:
        raise ValueError("splits must contain train and/or val definitions")
    rows: list[dict[str, Any]] = []
    aggregate_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}

    for split_name, split in splits.items():
        if not isinstance(split, Mapping):
            raise ValueError(f"Split {split_name} must be a mapping")
        catalog_path = _resolve(str(split["maps_file"]), base)
        catalog = _load_maps(catalog_path)
        map_names = _select_map_names(catalog, split)
        seeds = _integer_grid(split.get("seeds", []), "seeds")
        agent_counts = _integer_grid(split.get("num_agents", []), "num_agents")
        if not map_names or not seeds or not agent_counts:
            raise ValueError(f"Split {split_name} requires maps, seeds and num_agents")
        unknown = sorted(set(map_names) - set(catalog))
        if unknown:
            raise ValueError(f"Unknown maps in split {split_name}: {unknown}")

        family = str(split.get("family", split_name))
        subset = str(split.get("subset", split_name))
        if subset not in aggregate_records:
            raise ValueError(
                f"Split {split_name} has unsupported subset {subset!r}; "
                "dataset generation supports only 'train' and 'val'"
            )
        split_dir = output_root / str(split.get("output_dir", split_name))
        split_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        combinations: Iterable[tuple[str, int, int]] = product(map_names, seeds, agent_counts)
        if max_scenarios_per_split is not None:
            combinations = islice(combinations, max_scenarios_per_split)
        tasks = (
            {
                "episode_index": episode_index, "split": str(split_name),
                "family": family, "subset": subset, "map_name": map_name,
                "seed": seed, "num_agents": num_agents, "map_data": catalog[map_name],
                "planner_config": planner_config, "split_dir": str(split_dir),
                "output_root": str(output_root),
            }
            for episode_index, (map_name, seed, num_agents) in enumerate(combinations)
        )
        if workers == 1:
            results: Iterable[dict[str, Any]] = map(_run_scenario, tasks)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(processes=workers)
            results = pool.imap_unordered(_run_scenario, tasks, chunksize=1)
        try:
            for result in results:
                row = result["row"]
                rows.append(row)
                if result["record"] is not None:
                    records.append(result["record"])
                    aggregate_records[subset].append(result["aggregate_record"])
                    print(f"split={split_name} map={row['map_name']} seed={row['seed']} "
                          f"agents={row['num_agents']} makespan={row['makespan']} "
                          f"soc={row['soc']} samples={row['num_samples']}", flush=True)
                else:
                    print(f"FAILED split={split_name} map={row['map_name']} seed={row['seed']} "
                          f"agents={row['num_agents']}: {row['error']}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        records.sort(key=lambda record: record["path"])
        _write_manifest(records, split_dir / "manifest.jsonl")

    for subset_records in aggregate_records.values():
        subset_records.sort(key=lambda record: (record["split"], record["path"]))
    _write_manifest(aggregate_records.get("train", []), output_root / "train_manifest.jsonl")
    _write_manifest(aggregate_records.get("val", []), output_root / "val_manifest.jsonl")

    (output_root / "resolved_grid_config.yaml").write_text(
        yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
    )
    rows.sort(key=lambda row: (row["split"], int(row["episode_index"])))
    _write_results(rows, output_root)
    (output_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = _dataset_summary(rows)
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n[DatasetSummary]")
    print(json.dumps(summary, indent=2), flush=True)
    _print_views(rows, values.get("results_views", {}))
    return output_root


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate train/val MAPF datasets from a grid YAML")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-scenarios-per-split", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    print(f"Dataset root: {generate_grid_dataset(args.config, args.max_scenarios_per_split, args.workers)}")


if __name__ == "__main__":
    main()
