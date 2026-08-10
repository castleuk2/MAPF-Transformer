from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mapf_pct.config import ModelConfig
from mapf_pct.cpp import CppEpisodeFeatureGenerator
from mapf_pct.data.npz_dataset import EpisodeFeatureBuilder, _Episode


IGNORED = {
    "all_agent_actions", "action_soft_targets", "reason_labels",
    "reason_valid", "scene_risk_labels",
}
FLOAT_RTOL = 1.0e-5
FLOAT_ATOL = 1.0e-6


def selected_steps(arrival: int, time_steps: int, ratio: float) -> set[int]:
    result = set(range(arrival))
    suffix = max(0, time_steps - arrival)
    if suffix and ratio:
        keep = min(suffix, max(1, int(math.ceil(suffix * ratio))))
        result.update(map(int, arrival + np.linspace(0, suffix - 1, num=keep, dtype=np.int64)))
    return result


def load_records(manifest: Path, split: str) -> list[dict]:
    records = []
    for index, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        path = Path(record["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        records.append({**record, "path": str(path.resolve()), "split": split, "record_index": index})
    return records


def compare_episode(record: dict, ratio: float) -> dict:
    started = time.perf_counter()
    with np.load(record["path"], allow_pickle=False) as data:
        obstacles = np.asarray(data["obstacles"], dtype=np.uint8)
        positions = np.asarray(data["positions"], dtype=np.int64)
        goals = np.asarray(data["goals"], dtype=np.int64)
        actions = np.asarray(data["actions"], dtype=np.int64)
    if goals.ndim != 2:
        return {"ok": False, "unsupported": "dynamic goals", "record": record}
    cfg = ModelConfig()
    episode = _Episode(obstacles, positions, goals, actions)
    reference = EpisodeFeatureBuilder(cfg)
    cpp = CppEpisodeFeatureGenerator(cfg)
    arrivals = list(map(int, record["arrival_steps"]))
    time_steps = int(record["time_steps"])
    exposure = [selected_steps(arrival, time_steps, ratio) for arrival in arrivals]
    checked_samples = 0
    checked_tensors = 0
    max_abs_error: dict[str, float] = {}
    mismatch = None

    for step in range(time_steps):
        actual_batch = cpp.generate(obstacles, positions[step], goals)
        for ego in range(actions.shape[1]):
            if step not in exposure[ego]:
                continue
            expected = reference.build_episode(episode, step, ego)
            checked_samples += 1
            for name, expected_tensor in expected.tensors():
                if name in IGNORED:
                    continue
                actual_tensor = getattr(actual_batch, name)[ego]
                checked_tensors += 1
                if expected_tensor.is_floating_point():
                    error = float((actual_tensor - expected_tensor).abs().max())
                    max_abs_error[name] = max(max_abs_error.get(name, 0.0), error)
                    equal = torch.allclose(actual_tensor, expected_tensor, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
                else:
                    equal = torch.equal(actual_tensor, expected_tensor)
                if not equal:
                    mismatch = {
                        "field": name, "step": step, "ego": ego,
                        "expected_shape": list(expected_tensor.shape),
                        "actual_shape": list(actual_tensor.shape),
                        "different_values": int((actual_tensor != expected_tensor).sum())
                            if not expected_tensor.is_floating_point() else None,
                        "max_abs_error": float((actual_tensor.float() - expected_tensor.float()).abs().max()),
                    }
                    break
            if mismatch is not None:
                break
        if mismatch is not None:
            break
        cpp.commit_actions(actions[step])

    return {
        "ok": mismatch is None, "split": record["split"], "record_index": record["record_index"],
        "path": record["path"], "map_family": record.get("map_family"),
        "num_agents": int(record["num_agents"]), "time_steps": time_steps,
        "checked_samples": checked_samples, "checked_tensors": checked_tensors,
        "max_abs_error": max_abs_error, "mismatch": mismatch,
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustive Python/C++ feature parity over Train and Validation")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--goal-wait-keep-ratio", type=float, default=0.2)
    parser.add_argument("--max-records-per-split", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "episode_results.jsonl"
    completed: set[tuple[str, int]] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed.add((row["split"], int(row["record_index"])))
    train_records = load_records(args.train_manifest.resolve(), "train")
    val_records = load_records(args.val_manifest.resolve(), "val")
    if args.max_records_per_split is not None:
        train_records = train_records[: args.max_records_per_split]
        val_records = val_records[: args.max_records_per_split]
    records = train_records + val_records
    pending = [r for r in records if (r["split"], r["record_index"]) not in completed]
    print(f"records={len(records)} completed={len(completed)} pending={len(pending)} workers={args.workers}", flush=True)

    total_samples = total_tensors = failures = processed = 0
    wall_started = time.perf_counter()
    with results_path.open("a", encoding="utf-8") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compare_episode, record, args.goal_wait_keep_ratio): record for record in pending}
        for future in as_completed(futures):
            row = future.result()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            processed += 1
            total_samples += int(row.get("checked_samples", 0)); total_tensors += int(row.get("checked_tensors", 0))
            failures += int(not row["ok"])
            if processed == 1 or processed % 25 == 0 or not row["ok"]:
                rate = processed / max(1.0e-9, time.perf_counter() - wall_started)
                eta = (len(pending) - processed) / max(rate, 1.0e-9)
                print(f"processed={processed}/{len(pending)} samples={total_samples} failures={failures} rate={rate:.3f} ep/s eta_s={eta:.0f}", flush=True)

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        "episodes": len(rows), "passed_episodes": sum(int(r["ok"]) for r in rows),
        "failed_episodes": sum(int(not r["ok"]) for r in rows),
        "checked_samples": sum(int(r.get("checked_samples", 0)) for r in rows),
        "checked_tensors": sum(int(r.get("checked_tensors", 0)) for r in rows),
        "float_rtol": FLOAT_RTOL, "float_atol": FLOAT_ATOL,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
