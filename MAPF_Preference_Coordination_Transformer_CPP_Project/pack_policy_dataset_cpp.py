from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from mapf_pct.config import ModelConfig, load_config
from mapf_pct.constants import Action, DeltaCTG
from mapf_pct.cpp import CppEpisodeFeatureGenerator
from mapf_pct.data.packed_policy_dataset import PACKED_POLICY_VERSION, model_fingerprint
from mapf_pct.types import PolicyBatch
from pack_policy_dataset import storage_array


def _wait_times(arrival: int, time_steps: int, ratio: float) -> np.ndarray:
    suffix = max(0, time_steps - arrival)
    if suffix == 0 or ratio == 0:
        return np.empty(0, dtype=np.int64)
    keep = min(suffix, max(1, int(math.ceil(suffix * ratio))))
    return arrival + np.linspace(0, suffix - 1, num=keep, dtype=np.int64)


def _slice(batch: PolicyBatch, index: int) -> PolicyBatch:
    return PolicyBatch(**{
        field.name: None if getattr(batch, field.name) is None else getattr(batch, field.name)[index].clone()
        for field in fields(PolicyBatch)
    })


def _attach_supervision(
    batch: PolicyBatch, slot_ids: torch.Tensor, expert_actions: np.ndarray, cfg: ModelConfig
) -> None:
    valid = batch.agent_valid
    safe_ids = slot_ids.clamp_min(0)
    labels = torch.from_numpy(np.asarray(expert_actions, dtype=np.int64))[safe_ids]
    labels = labels.masked_fill(~valid, int(Action.WAIT))

    reasons = torch.zeros(labels.shape[0], cfg.max_agents, cfg.reason_classes)
    chosen_delta = batch.candidate_delta_ctg.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    chosen_bottleneck = batch.candidate_bottleneck.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    reasons[:, :, 0] = (chosen_delta == int(DeltaCTG.DECREASE)) & valid
    reasons[:, :, 1] = batch.on_goal & (labels == int(Action.WAIT)) & valid
    reasons[:, :, 5] = (
        (labels == int(Action.WAIT)) & (batch.candidate_contenders.amax(2) > 0) & valid
    )
    reasons[:, :, 6] = chosen_bottleneck & valid
    reasons[:, :, 9] = (
        ((batch.history_selected == int(Action.WAIT)).sum(2) >= 3)
        & (labels != int(Action.WAIT)) & valid
    )
    risk_score = batch.scene_numeric[:, 11] + batch.scene_numeric[:, 12]
    risk = torch.where(risk_score > .35, 2, torch.where(risk_score > .12, 1, 0)).long()

    batch.all_agent_actions = labels.long()
    batch.action_soft_targets = None
    batch.reason_labels = reasons
    batch.reason_valid = valid.clone()
    batch.scene_risk_labels = risk


def _selections(record: dict, ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "index_path" in record:
        with np.load(record["index_path"], allow_pickle=False) as selector:
            times = np.asarray(selector["time"], dtype=np.int64)
            egos = np.asarray(selector["ego"], dtype=np.int64)
        return times, egos, np.asarray([len(times)], dtype=np.int64)
    arrivals = np.asarray(record["arrival_steps"], dtype=np.int64)
    time_steps = int(record["time_steps"])
    times: list[int] = []
    egos: list[int] = []
    counts: list[int] = []
    for ego, arrival in enumerate(arrivals):
        selected = list(range(int(arrival))) + _wait_times(int(arrival), time_steps, ratio).tolist()
        times.extend(selected)
        egos.extend([ego] * len(selected))
        counts.append(len(selected))
    return np.asarray(times), np.asarray(egos), np.asarray(counts, dtype=np.int64)


def convert(task):
    record, output_raw, ratio, overwrite, model_dict, data_dict = task
    indexed = "index_path" in record
    source = Path(record["episode_path"] if indexed else record["path"]).resolve()
    identity = str(source) + ("|" + str(record["index_path"]) if indexed else "")
    digest = hashlib.blake2b(identity.encode(), digest_size=8).hexdigest()
    output = Path(output_raw) / "episodes" / f"{digest}_{source.name}"
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_time, selected_ego, counts = _selections(record, ratio)
    if output.exists() and not overwrite:
        return output, counts, output.stat().st_size, True

    with np.load(source, allow_pickle=False) as archive:
        obstacles = np.asarray(archive["obstacles"], dtype=np.uint8)
        positions = np.asarray(archive["positions"], dtype=np.int64)
        goals = np.asarray(archive["goals"], dtype=np.int64)
        actions = np.asarray(archive["actions"], dtype=np.int64)
    if goals.ndim != 2:
        raise ValueError(f"C++ packer currently requires static goals [N,2]: {source}")
    if len(selected_time) == 0:
        raise ValueError(f"episode selection is empty: {source}")

    by_time: dict[int, list[tuple[int, int]]] = {}
    for output_index, (step, ego) in enumerate(zip(selected_time, selected_ego)):
        by_time.setdefault(int(step), []).append((output_index, int(ego)))
    if int(selected_time.min()) < 0 or int(selected_time.max()) >= actions.shape[0]:
        raise IndexError(f"selection time outside episode: {source}")
    if int(selected_ego.min()) < 0 or int(selected_ego.max()) >= actions.shape[1]:
        raise IndexError(f"selection ego outside episode: {source}")

    cfg = ModelConfig(**model_dict)
    generator = CppEpisodeFeatureGenerator(
        cfg, task_mode=data_dict["task_mode"], target_mode=data_dict["target_mode"]
    )
    samples: list[PolicyBatch | None] = [None] * len(selected_time)
    for step in range(int(selected_time.max()) + 1):
        batch, slot_ids = generator.generate_with_slot_ids(obstacles, positions[step], goals)
        _attach_supervision(batch, slot_ids, actions[step], cfg)
        for output_index, ego in by_time.get(step, ()):
            samples[output_index] = _slice(batch, ego)
        generator.commit_actions(actions[step])
    if any(sample is None for sample in samples):
        raise AssertionError(f"not all selected samples were generated: {source}")

    arrays = {}
    typed_samples = [sample for sample in samples if sample is not None]
    for field in fields(PolicyBatch):
        values = [getattr(sample, field.name) for sample in typed_samples]
        if values[0] is None:
            continue
        original_dtype = str(values[0].dtype).removeprefix("torch.")
        original = torch.stack(values)
        stored = storage_array(original)
        if not torch.equal(original, torch.from_numpy(stored).to(original.dtype)):
            raise AssertionError(f"lossy packed conversion detected: {field.name}")
        arrays[field.name] = stored
        arrays[f"__dtype__{field.name}"] = np.asarray(original_dtype)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, output)
    return output, counts, output.stat().st_size, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MPCT PolicyBatch features with the C++ generator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ratio = config.data.goal_wait_keep_ratio
    manifest = args.manifest.resolve()
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit_episodes is not None:
        records = records[:args.limit_episodes]
    for record in records:
        path_key = "episode_path" if "episode_path" in record else "path"
        path = Path(record[path_key])
        if not path.is_absolute():
            record[path_key] = str((manifest.parent / path).resolve())
        if "index_path" in record:
            index_path = Path(record["index_path"])
            if not index_path.is_absolute():
                record["index_path"] = str((manifest.parent / index_path).resolve())

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    tasks = [
        (record, str(output), ratio, args.overwrite, config_dict["model"], config_dict["data"])
        for record in records
    ]
    converted = []
    total = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (record, result) in enumerate(zip(records, pool.map(convert, tasks, chunksize=1)), 1):
            path, counts, size, was_skipped = result
            converted.append({
                "path": os.path.relpath(path, output), "samples": int(counts.sum()),
                "sample_counts": counts.tolist(), "map_family": record.get("map_family"),
                "num_agents": int(record.get("num_agents", len(counts))),
                "packed_policy_version": PACKED_POLICY_VERSION,
                "model_fingerprint": model_fingerprint(config.model),
                "exact_roundtrip_verified": True,
                "feature_generator": "cpp",
            })
            total += size
            skipped += int(was_skipped)
            if i == 1 or i % 100 == 0 or i == len(records):
                print(f"packed={i}/{len(records)} skipped={skipped} size_gib={total/1024**3:.3f}", flush=True)
    target = output / "manifest.jsonl"
    target.write_text("".join(json.dumps(record) + "\n" for record in converted), encoding="utf-8")
    metadata = {
        "source_manifest": str(manifest), "episodes": len(converted),
        "samples": sum(record["samples"] for record in converted), "bytes": total,
        "goal_wait_keep_ratio": ratio, "model_fingerprint": model_fingerprint(config.model),
        "packed_policy_version": PACKED_POLICY_VERSION, "feature_generator": "cpp",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
