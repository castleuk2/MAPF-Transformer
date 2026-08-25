from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def inspect(path: Path, wait_ratio: float, check_files: bool) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    total_samples, missing = 0, []
    strata = Counter()
    for record in records:
        episode = Path(record["path"])
        if not episode.is_absolute():
            episode = path.parent / episode
        if check_files and not episode.is_file():
            missing.append(str(episode))
        arrivals = np.asarray(record["arrival_steps"], dtype=np.int64)
        time_steps = int(record["time_steps"])
        suffix = np.maximum(0, time_steps - arrivals)
        waits = np.where(suffix > 0, np.maximum(1, np.ceil(suffix * wait_ratio)).astype(np.int64), 0)
        total_samples += int((arrivals + waits).sum())
        strata[(str(record.get("map_family", "unknown")), int(record.get("num_agents", len(arrivals))))] += 1
    return {
        "manifest": str(path.resolve()), "episodes": len(records), "policy_samples": total_samples,
        "goal_wait_keep_ratio": wait_ratio, "missing_files": len(missing),
        "missing_examples": missing[:10],
        "episode_strata": {f"{family}/n{agents}": count for (family, agents), count in sorted(strata.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--goal-wait-keep-ratio", type=float, default=0.2)
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()
    reports = [
        inspect(args.train_manifest.resolve(), args.goal_wait_keep_ratio, not args.skip_file_check),
        inspect(args.val_manifest.resolve(), args.goal_wait_keep_ratio, not args.skip_file_check),
    ]
    print(json.dumps(reports, indent=2))
    if any(report["missing_files"] for report in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
