from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from mapf_pct.config import ModelConfig
from mapf_pct.cpp import CppEpisodeFeatureGenerator
from mapf_pct.data.npz_dataset import EpisodeFeatureBuilder, _Episode
from mapf_pct.types import stack_policy_batches


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = ModelConfig()
    results = []
    for path in args.episodes:
        with np.load(path, allow_pickle=False) as data:
            obstacles = np.asarray(data["obstacles"], dtype=np.uint8)
            positions = np.asarray(data["positions"], dtype=np.int64)
            goals = np.asarray(data["goals"], dtype=np.int64)
            actions = np.asarray(data["actions"], dtype=np.int64)
        if goals.ndim != 2:
            raise ValueError("benchmark currently requires static [N,2] goals")
        steps = min(args.steps, actions.shape[0])

        cpp_started = time.perf_counter()
        cpp = CppEpisodeFeatureGenerator(cfg)
        cpp._ensure(obstacles, goals)
        cpp_init = time.perf_counter() - cpp_started
        python = EpisodeFeatureBuilder(cfg)
        cpp_times, python_times = [], []
        for step in range(steps):
            started = time.perf_counter()
            cpp.generate(obstacles, positions[step], goals)
            cpp_times.append(time.perf_counter() - started)
            cpp.commit_actions(actions[step])

            episode = _Episode(obstacles, positions[: step + 2], goals, actions[: step + 1])
            started = time.perf_counter()
            stack_policy_batches([python.build_episode(episode, step, ego) for ego in range(actions.shape[1])])
            python_times.append(time.perf_counter() - started)

        def stats(values: list[float]) -> dict[str, float]:
            return {
                "mean_ms": 1000 * statistics.mean(values),
                "median_ms": 1000 * statistics.median(values),
                "p95_ms": 1000 * percentile(values, 95),
                "steps_per_second": 1.0 / statistics.mean(values),
            }
        row = {
            "episode": str(path.resolve()), "agents": int(actions.shape[1]), "steps": steps,
            "cpp_initialization_ms": 1000 * cpp_init,
            "python": stats(python_times), "cpp": stats(cpp_times),
            "speedup": statistics.mean(python_times) / statistics.mean(cpp_times),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))
    payload = {"results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
