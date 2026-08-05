from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mapf_map_transformer.policy_data import PolicyHistoryHaloDataset


def layout_hash(path: str) -> str:
    with np.load(path, allow_pickle=False) as archive:
        obstacles = np.ascontiguousarray(archive["obstacles"], dtype=np.uint8)
    return hashlib.sha256(obstacles.tobytes() + str(obstacles.shape).encode()).hexdigest()


def exposure_count(dataset: PolicyHistoryHaloDataset) -> int:
    total = 0
    for index in range(len(dataset)):
        _, frame, _ = dataset.resolve_index(index)
        available = min(dataset.history_frames, frame + 1)
        if dataset.history_augmentation:
            low = min(dataset.min_history_frames, available)
            rng = np.random.default_rng(dataset.seed + index * 104729)
            total += int(rng.integers(low, available + 1))
        else:
            total += available
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/policy_exposure_17x17"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--crop-checks", type=int, default=100)
    args = parser.parse_args()
    result: dict[str, dict] = {}
    layouts: dict[str, set[str]] = {}
    for split, augmented, seed in (("train", True, 42), ("val", False, 43), ("eval", False, 44)):
        manifest = args.data_dir / split / "manifest.jsonl"
        dataset = PolicyHistoryHaloDataset(manifest, history_augmentation=augmented, seed=seed)
        hashes = {layout_hash(record["source_path"]) for record in dataset.records}
        layouts[split] = hashes
        family_episodes = {"maze": 0, "random": 0}
        for record in dataset.records:
            family_episodes["maze" if record["map_family_id"] == 0 else "random"] += 1
        result[split] = {
            "episodes": len(dataset.records),
            "policy_samples": len(dataset),
            "valid_map_exposures": exposure_count(dataset),
            "layout_hashes": len(hashes),
            "family_episodes": family_episodes,
            "history_augmentation": augmented,
        }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "eval"), ("val", "eval")):
        count = len(layouts[left] & layouts[right])
        overlaps[f"{left}_{right}"] = count
        if count:
            raise AssertionError(f"Layout leakage between {left} and {right}: {count}")
    result["layout_overlap_counts"] = overlaps
    output = args.output or args.data_dir / "validation_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"summary={output}")


if __name__ == "__main__":
    main()
