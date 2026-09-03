#!/usr/bin/env python3
"""Create a fixed-size, lossless JIT-Yield v2 indexed view from large LNS2 data.

The scanner keeps bounded reservoirs instead of materialising all ~125M sample
indices.  It preserves a local temporal window around every selected Goal-leave
event and fails by default if a requested marginal would require repetition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TARGET = {"off_goal": .7693051713345314, "final_goal_wait": .17068104719019445,
          "pre_final_goal_wait": .05012822324980312, "goal_leave": .009885558225470993}
DELAY = {"1": .9268734839780416, "2-3": .034469551895825354,
         "4-5": .017617770968977404, "6-10": .013455891740074046,
         ">10": .007583301417081297}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def wait_times(arrival: int, steps: int, ratio: float = .2) -> np.ndarray:
    suffix = max(0, steps - arrival)
    if not suffix:
        return np.empty(0, dtype=np.int32)
    keep = min(suffix, max(1, int(math.ceil(suffix * ratio))))
    return (arrival + np.linspace(0, suffix - 1, num=keep, dtype=np.int64)).astype(np.int32)


def packed_mapping(arrivals: np.ndarray, steps: int, counts: list[int]):
    mapping, reverse, offset = {}, [], 0
    for ego, (arrival, count) in enumerate(zip(arrivals.tolist(), counts)):
        times = np.concatenate((np.arange(arrival, dtype=np.int32), wait_times(arrival, steps)))
        if len(times) != int(count):
            raise ValueError(f"packed sample-count mismatch: {len(times)} != {count}")
        for local, time in enumerate(times.tolist()):
            mapping[(time, ego)] = offset + local
            reverse.append((time, ego))
        offset += len(times)
    return mapping, reverse


def prepare(raw_manifest: Path, packed_manifest: Path, max_episodes: int | None = None):
    raw_manifest, packed_manifest = raw_manifest.resolve(), packed_manifest.resolve()
    raw, packed = read_jsonl(raw_manifest), read_jsonl(packed_manifest)
    if len(raw) != len(packed):
        raise ValueError("raw/packed episode count mismatch")
    if max_episodes:
        raw, packed = raw[:max_episodes], packed[:max_episodes]
    records = []
    for ordinal, (source, pack) in enumerate(zip(raw, packed)):
        source_path = resolve(raw_manifest.parent, source["path"])
        packed_path = resolve(packed_manifest.parent, pack["path"])
        if not packed_path.name.endswith(source_path.name):
            raise ValueError(f"episode order mismatch: {source_path.name} {packed_path.name}")
        records.append({"ordinal": ordinal, "raw_path": source_path, "packed_path": packed_path,
                        "sample_counts": list(map(int, pack["sample_counts"])),
                        "samples": int(pack["samples"]),
                        "group": (source["map_family"], int(source["num_agents"]))})
    return records


def largest_remainder(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: total * ratio for key, ratio in ratios.items()}
    out = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda item: raw[item] - out[item], reverse=True)[:total - sum(out.values())]:
        out[key] += 1
    return out


def delay_bin(value: int | None) -> str:
    if value == 1: return "1"
    if value is not None and value <= 3: return "2-3"
    if value is not None and value <= 5: return "4-5"
    if value is not None and value <= 10: return "6-10"
    return ">10"


def priority(seed: int, episode: int, index: int, salt: str) -> int:
    payload = f"{seed}:{episode}:{index}:{salt}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def classify(record: dict):
    with np.load(record["raw_path"], allow_pickle=False) as archive:
        positions = np.asarray(archive["positions"])
        goals = np.asarray(archive["goals"])
        arrivals = np.asarray(archive["arrival_steps"])
    if goals.ndim == 3:
        goals = goals[-1]
    steps = len(positions) - 1
    mapping, reverse = packed_mapping(arrivals, steps, record["sample_counts"])
    at_goal = np.all(positions == goals[None], axis=2)
    moved = np.any(positions[1:] != positions[:-1], axis=2)
    categories, delays = {}, {}
    for index, (time, ego) in enumerate(reverse):
        if time >= arrivals[ego]:
            category = "final_goal_wait"
        elif at_goal[time, ego] and not moved[time, ego]:
            category = "pre_final_goal_wait"
        elif at_goal[time, ego] and moved[time, ego]:
            category = "goal_leave"
            other = np.arange(positions.shape[1]) != ego
            delay = None
            for future in range(time + 1, len(positions)):
                if np.any(np.all(positions[future, other] == goals[ego], axis=1)):
                    delay = future - time
                    break
            delays[index] = delay_bin(delay)
        else:
            category = "off_goal"
        categories[index] = category
    return positions, goals, arrivals, mapping, reverse, categories, delays


def census(records: list[dict]):
    available, available_leave = Counter(), Counter()
    for episode, record in enumerate(records):
        _, _, _, _, _, categories, delays = classify(record)
        available.update(categories.values())
        available_leave.update(delays.values())
        if (episode + 1) % 1000 == 0 or episode + 1 == len(records):
            print(f"census {episode + 1}/{len(records)}", flush=True)
    return available, available_leave


def feasible_target(target: int, available: Counter, available_leave: Counter) -> bool:
    quotas = largest_remainder(target, TARGET)
    leave_quotas = largest_remainder(quotas["goal_leave"], DELAY)
    return (all(quotas[key] <= available[key] for key in quotas)
            and all(leave_quotas[key] <= available_leave[key] for key in leave_quotas))


def maximum_unique_target(available: Counter, available_leave: Counter) -> int:
    bounds = [int(available[key] / ratio) for key, ratio in TARGET.items() if ratio > 0]
    for key, ratio in DELAY.items():
        combined = TARGET["goal_leave"] * ratio
        if combined > 0:
            bounds.append(int(available_leave[key] / combined))
    low, high = 0, max(0, min(bounds))
    while low < high:
        middle = (low + high + 1) // 2
        if feasible_target(middle, available, available_leave):
            low = middle
        else:
            high = middle - 1
    return low


def build(records: list[dict], target: int | str, seed: int, allow_repeat: bool):
    used_auto = target == "auto"
    if used_auto:
        census_counts, census_leave = census(records)
        target = maximum_unique_target(census_counts, census_leave)
        if target <= 0:
            raise RuntimeError("the source cannot form a non-empty ratio-matched Dataset")
        print(f"maximum unique ratio-matched samples={target}", flush=True)
    target = int(target)
    quotas = largest_remainder(target, TARGET)
    leave_quotas = largest_remainder(quotas["goal_leave"], DELAY)
    # Keep deterministic smallest-hash candidates.  Capacity leaves ample room
    # for keys already introduced by temporal context windows.
    capacity = {key: value + min(target, 100_000) for key, value in quotas.items()}
    reservoirs: dict[str, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    leave_reservoirs: dict[str, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    available, available_leave = Counter(), Counter()
    import heapq

    def offer(heap, cap, score, key):
        if cap <= 0:
            return
        item = (-score, key)
        if len(heap) < cap: heapq.heappush(heap, item)
        elif score < -heap[0][0]: heapq.heapreplace(heap, item)

    for episode, record in enumerate(records):
        _, _, _, _, _, categories, delays = classify(record)
        for index, category in categories.items():
            key = (episode, index); available[category] += 1
            offer(reservoirs[category], capacity[category], priority(seed, episode, index, category), key)
            if category == "goal_leave":
                bucket = delays[index]; available_leave[bucket] += 1
                offer(leave_reservoirs[bucket], leave_quotas[bucket], priority(seed, episode, index, bucket), key)
        if (episode + 1) % 1000 == 0 or episode + 1 == len(records):
            print(f"scan {episode + 1}/{len(records)}", flush=True)

    missing = {key: quotas[key] - available[key] for key in quotas if available[key] < quotas[key]}
    missing.update({f"leave:{key}": leave_quotas[key] - available_leave[key]
                    for key in leave_quotas if available_leave[key] < leave_quotas[key]})
    if missing and not allow_repeat:
        raise RuntimeError(f"unique samples are insufficient; rerun with --allow-repeat: {missing}")

    centers = {key for heap in leave_reservoirs.values() for _, key in heap}
    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    categories_by_key: dict[tuple[int, int], str] = {}
    for episode in sorted({key[0] for key in centers}):
        positions, _, _, mapping, reverse, categories, _ = classify(records[episode])
        center_indices = {index for ep, index in centers if ep == episode}
        all_leaves = {index for index, category in categories.items() if category == "goal_leave"}
        for center_index in center_indices:
            time, ego = reverse[center_index]
            with np.load(records[episode]["raw_path"], allow_pickle=False) as archive:
                goals = np.asarray(archive["goals"])
            if goals.ndim == 3: goals = goals[-1]
            ego_at_goal = np.all(positions[:, ego] == goals[ego], axis=1)
            returns = np.flatnonzero(ego_at_goal[time + 1:])
            return_time = time + 1 + int(returns[0])
            for current in range(max(0, time - 5), min(len(positions) - 1, return_time + 6)):
                packed_index = mapping.get((current, ego))
                if packed_index is None or (packed_index in all_leaves and packed_index not in center_indices):
                    continue
                key = (episode, packed_index)
                if key not in selected_set:
                    selected.append(key); selected_set.add(key); categories_by_key[key] = categories[packed_index]

    counts = Counter(categories_by_key.values())
    repeated = Counter()
    for category, wanted in quotas.items():
        need = wanted - counts[category]
        if need < 0:
            raise RuntimeError(f"context exceeds {category} quota: {counts[category]} > {wanted}")
        pool = [key for _, key in sorted(reservoirs[category], reverse=True) if key not in selected_set]
        if len(pool) < need and not allow_repeat:
            raise RuntimeError(f"reservoir shortage for {category}: {len(pool)} < {need}")
        chosen = pool[:need]
        if len(chosen) < need:
            source = pool or [key for _, key in reservoirs[category]]
            chosen.extend(source[index % len(source)] for index in range(need - len(chosen)))
            repeated[category] += need - len(pool)
        selected.extend(chosen); selected_set.update(chosen); counts[category] += len(chosen)

    by_episode: dict[int, list[int]] = defaultdict(list)
    for episode, index in selected:
        by_episode[episode].append(index)
    return by_episode, {"target_samples": target,
                        "selection_mode": "maximum_unique" if used_auto else "fixed_size",
                        "target_category_counts": quotas,
                        "target_leave_delay_counts": leave_quotas,
                        "available_unique_category_counts": dict(available),
                        "available_unique_leave_delay_counts": dict(available_leave),
                        "actual_category_counts": dict(counts), "repeated_samples": dict(repeated),
                        "selected_leave_events": len(centers),
                        "context_preserved_around_selected_events": True}


def write_view(root: Path, records: list[dict], selected: dict[int, list[int]], metadata: dict):
    root.mkdir(parents=True, exist_ok=True); index_dir = root / "indices"; index_dir.mkdir(exist_ok=True)
    rows = []
    for episode, indices in sorted(selected.items()):
        index_path = index_dir / f"index_{episode:07d}.npz"
        np.savez_compressed(index_path, index=np.asarray(sorted(indices), dtype=np.int64))
        record = records[episode]
        rows.append({"mode": "packed_indexed", "packed_path": str(record["packed_path"]),
                     "index_path": str(index_path.resolve()), "samples": len(indices),
                     "map_family": record["group"][0], "num_agents": record["group"][1]})
    (root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata = {**metadata, "episodes_with_samples": len(rows), "samples": sum(row["samples"] for row in rows),
                "storage": "lossless indexed view of C++ precomputed Packed PolicyBatch"}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--packed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", required=True, help="integer sample count or 'auto' for maximum unique size")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--max-episodes", type=int, help="smoke-test only")
    args = parser.parse_args()
    records = prepare(args.raw_manifest, args.packed_manifest, args.max_episodes)
    sample_request: int | str = args.samples if args.samples == "auto" else int(args.samples)
    selected, metadata = build(records, sample_request, args.seed, args.allow_repeat)
    write_view(args.output, records, selected, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
