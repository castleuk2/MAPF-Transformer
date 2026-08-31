#!/usr/bin/env python3
"""Regenerate an existing MAPF manifest with LaCAM3 as the sole changed factor."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pogema_mapf_transformer.episode_io import build_episode_data, save_episode
from pogema_mapf_transformer.expert import PrioritizedTimeExpandedPlanner


MOVES_TO_ACTION = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
_LIB = None


def _load_library(path: str):
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(path)
        _LIB.run_lacam.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
        _LIB.run_lacam.restype = ctypes.c_char_p
    return _LIB


def _movingai_inputs(obstacles: np.ndarray, starts: np.ndarray, goals: np.ndarray):
    height, width = obstacles.shape
    rows = ["".join("@" if value else "." for value in row) for row in obstacles]
    map_text = f"type octile\nheight {height}\nwidth {width}\nmap\n" + "\n".join(rows)
    scenario = ["version 1"]
    for idx, (start, goal) in enumerate(zip(starts, goals)):
        sy, sx = map(int, start)
        gy, gx = map(int, goal)
        scenario.append(f"{idx}\ttmp.map\t{width}\t{height}\t{sx}\t{sy}\t{gx}\t{gy}\t1")
    return map_text, "\n".join(scenario) + "\n"


def _parse_positions(payload: str, num_agents: int) -> np.ndarray:
    frames = []
    for line in payload.strip().splitlines():
        xy = [tuple(map(int, item.split(","))) for item in line.split("|") if item]
        if len(xy) != num_agents:
            raise ValueError(f"LaCAM3 returned {len(xy)} agents, expected {num_agents}")
        frames.append([(y, x) for x, y in xy])
    if not frames:
        raise ValueError("LaCAM3 returned an empty trajectory")
    return np.asarray(frames, dtype=np.int16)


def _positions_to_actions(positions: np.ndarray) -> np.ndarray:
    delta = positions[1:].astype(np.int32) - positions[:-1].astype(np.int32)
    actions = np.empty(delta.shape[:2], dtype=np.uint8)
    for t in range(delta.shape[0]):
        for agent in range(delta.shape[1]):
            key = tuple(map(int, delta[t, agent]))
            if key not in MOVES_TO_ACTION:
                raise ValueError(f"Non-cardinal transition at t={t}, agent={agent}: {key}")
            actions[t, agent] = MOVES_TO_ACTION[key]
    return actions


def _solve_one(task):
    index, record, source_root, output_root, lib_path, timeout, max_steps, overwrite = task
    source_path = Path(source_root) / record["path"]
    relative_path = Path(record["path"])
    if relative_path.is_absolute():
        relative_path = Path(record.get("map_family", "episodes")) / relative_path.name
    output_path = Path(output_root) / relative_path
    try:
        if output_path.exists() and not overwrite:
            with np.load(output_path, allow_pickle=False) as data:
                obstacles = np.asarray(data["obstacles"], dtype=np.uint8)
                positions = np.asarray(data["positions"], dtype=np.int16)
                goals = np.asarray(data["goals"], dtype=np.int16)
                actions = np.asarray(data["actions"], dtype=np.uint8)
                metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
            episode = build_episode_data(obstacles, positions, goals, actions, metadata=metadata)
            arrivals = episode.get_arrival_steps()
            result = dict(record)
            result.update({
                "path": str(relative_path), "planner": "LaCAM3",
                "time_steps": int(len(actions)), "makespan": int(len(actions)),
                "arrival_steps": arrivals.tolist(), "num_samples": int(arrivals.sum()),
                "soc": int(arrivals.sum()),
                "runtime_sec": float(metadata.get("expert_runtime_sec", 0.0)),
                "success": True, "resumed": True,
            })
            return index, result, None
        with np.load(source_path, allow_pickle=False) as data:
            obstacles = np.asarray(data["obstacles"], dtype=np.uint8)
            starts = np.asarray(data["positions"][0], dtype=np.int16)
            goals = np.asarray(data["goals"], dtype=np.int16)
            source_metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        map_text, scenario_text = _movingai_inputs(obstacles, starts, goals)
        started = time.perf_counter()
        raw = _load_library(lib_path).run_lacam(
            map_text.encode(), scenario_text.encode(), len(starts), ctypes.c_float(timeout)
        )
        runtime = time.perf_counter() - started
        if not raw:
            raise RuntimeError("LaCAM3 returned a null response")
        payload = raw.decode("utf-8")
        if "ERROR" in payload:
            raise RuntimeError(payload.strip())
        positions = _parse_positions(payload, len(starts))
        actions = _positions_to_actions(positions)
        if len(actions) > max_steps:
            raise RuntimeError(f"makespan {len(actions)} exceeds max_steps {max_steps}")
        validated = PrioritizedTimeExpandedPlanner.validate_plan(starts, actions, obstacles)
        if not np.array_equal(validated, positions):
            raise RuntimeError("LaCAM3 positions disagree with action replay")
        if not np.array_equal(positions[-1], goals):
            raise RuntimeError("LaCAM3 trajectory does not finish at all goals")
        metadata = dict(source_metadata)
        metadata.update({
            "planner": "LaCAM3",
            "expert_runtime_sec": runtime,
            "expert_timeout_sec": timeout,
            "expert_max_steps": max_steps,
            "source_episode": str(source_path),
        })
        episode = build_episode_data(obstacles, positions, goals, actions, metadata=metadata)
        save_episode(output_path, episode)
        arrivals = episode.get_arrival_steps()
        result = dict(record)
        result.update({
            "path": str(relative_path),
            "planner": "LaCAM3",
            "time_steps": int(len(actions)),
            "makespan": int(len(actions)),
            "arrival_steps": arrivals.tolist(),
            "num_samples": int(arrivals.sum()),
            "soc": int(arrivals.sum()),
            "runtime_sec": runtime,
            "success": True,
        })
        return index, result, None
    except Exception as exc:
        return index, None, {"index": index, "path": str(relative_path), "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    source_root = Path(args.source_root).resolve() if args.source_root else manifest.parent
    output_root = Path(args.output_root).resolve()
    output_manifest = Path(args.output_manifest).resolve() if args.output_manifest else output_root / manifest.name
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        records = records[: args.limit]
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (i, record, str(source_root), str(output_root), str(Path(args.lib).resolve()),
         args.timeout, args.max_steps, args.overwrite)
        for i, record in enumerate(records)
    ]
    results = [None] * len(tasks)
    failures = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_solve_one, task) for task in tasks]
        for future in as_completed(futures):
            index, result, failure = future.result()
            results[index] = result
            if failure:
                failures.append(failure)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"completed={completed}/{len(tasks)} failures={len(failures)}", flush=True)
    successful = [record for record in results if record is not None]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text("".join(json.dumps(record) + "\n" for record in successful))
    failure_path = output_manifest.with_suffix(".failures.json")
    failure_path.write_text(json.dumps(failures, indent=2))
    summary = {
        "input_episodes": len(records), "successful": len(successful), "failed": len(failures),
        "sr": len(successful) / len(records) if records else 0.0,
        "mean_soc": float(np.mean([r["soc"] for r in successful])) if successful else None,
        "mean_makespan": float(np.mean([r["makespan"] for r in successful])) if successful else None,
        "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in successful])) if successful else None,
        "timeout_sec": args.timeout, "max_steps": args.max_steps,
    }
    output_manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
