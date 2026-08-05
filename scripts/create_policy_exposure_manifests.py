from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def records(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def resolve_source(record: dict, manifest: Path) -> Path:
    source = Path(record.get("source_path", record["path"]))
    return source if source.is_absolute() else (manifest.parent / source).resolve()


def enrich(record: dict, manifest: Path, subset: str) -> dict:
    source = resolve_source(record, manifest)
    with np.load(source, allow_pickle=False) as archive:
        positions = archive["positions"]
        goals = archive["goals"]
        time_steps = int(record.get("time_steps", len(positions) - 1))
        arrivals = record.get("arrival_steps")
        if arrivals is None:
            arrivals = []
            for agent, goal in enumerate(goals):
                hits = np.flatnonzero(np.all(positions[: time_steps + 1, agent] == goal, axis=-1))
                arrivals.append(int(hits[0]) if len(hits) else time_steps)
    return {
        "path": str(source),
        "source_path": str(source),
        "subset": subset,
        "split": record.get("split", subset),
        "map_family": record.get("map_family", "random"),
        "map_name": record.get("map_name", source.stem),
        "num_agents": int(record["num_agents"]),
        "time_steps": time_steps,
        "arrival_steps": [int(value) for value in arrivals],
        "planner": record.get("planner", "mapf_lns2"),
        "seed": int(record.get("seed", 0)),
    }


def write_split(source: Path, output: Path, subset: str, limits: dict[str, int] | None = None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        family_counts: dict[str, int] = {}
        for record in records(source):
            family = str(record.get("map_family", "random")).lower()
            if limits is not None and family_counts.get(family, 0) >= limits.get(family, 0):
                continue
            stream.write(json.dumps(enrich(record, source, subset), ensure_ascii=False) + "\n")
            family_counts[family] = family_counts.get(family, 0) + 1
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-maze-episodes", type=int, default=None)
    parser.add_argument("--eval-random-episodes", type=int, default=None)
    args = parser.parse_args()
    for name, source in (("train", args.train), ("val", args.val), ("eval", args.eval)):
        output = args.output_dir / name / "manifest.jsonl"
        limits = None
        if name == "eval" and args.eval_maze_episodes is not None and args.eval_random_episodes is not None:
            limits = {"maze": args.eval_maze_episodes, "random": args.eval_random_episodes}
        count = write_split(source.resolve(), output, name, limits)
        print(f"{name}: episodes={count:,} manifest={output}")


if __name__ == "__main__":
    main()
