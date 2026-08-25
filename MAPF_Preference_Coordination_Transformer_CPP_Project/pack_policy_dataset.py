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

from mapf_pct.config import load_config
from mapf_pct.data.npz_dataset import EpisodeFeatureBuilder
from mapf_pct.data.packed_policy_dataset import PACKED_POLICY_VERSION, model_fingerprint
from mapf_pct.types import PolicyBatch


def storage_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.dtype.kind in "iu":
        minimum, maximum = int(array.min(initial=0)), int(array.max(initial=0))
        for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32):
            info = np.iinfo(dtype)
            if info.min <= minimum and maximum <= info.max:
                packed = array.astype(dtype)
                if np.array_equal(packed.astype(array.dtype), array):
                    return packed
    return array.copy()


def wait_times(arrival: int, time_steps: int, ratio: float) -> np.ndarray:
    suffix = max(0, time_steps - arrival)
    if suffix == 0 or ratio == 0:
        return np.empty(0, dtype=np.int64)
    keep = min(suffix, max(1, int(math.ceil(suffix * ratio))))
    return arrival + np.linspace(0, suffix - 1, num=keep, dtype=np.int64)


def convert(task):
    record, output_raw, ratio, overwrite, model_dict, data_dict = task
    from mapf_pct.config import ModelConfig

    source = Path(record["path"]).resolve()
    digest = hashlib.blake2b(str(source).encode(), digest_size=8).hexdigest()
    output = Path(output_raw) / "episodes" / f"{digest}_{source.name}"
    output.parent.mkdir(parents=True, exist_ok=True)
    arrivals = np.asarray(record["arrival_steps"], dtype=np.int64)
    time_steps = int(record["time_steps"])
    waits = [wait_times(int(a), time_steps, ratio) for a in arrivals]
    counts = np.asarray([int(a) + len(w) for a, w in zip(arrivals, waits)], dtype=np.int64)
    if output.exists() and not overwrite:
        return output, counts, output.stat().st_size, True
    model = ModelConfig(**model_dict)
    builder = EpisodeFeatureBuilder(
        model,
        coordinate_order=data_dict["coordinate_order"],
        task_mode=data_dict["task_mode"],
        target_mode=data_dict["target_mode"],
    )
    samples = []
    for ego, arrival in enumerate(arrivals):
        for step in range(int(arrival)):
            samples.append(builder.build(source, step, ego))
        for step in waits[ego]:
            samples.append(builder.build(source, int(step), ego))
    if not samples:
        raise ValueError(f"episode produced no policy samples: {source}")
    arrays = {}
    for field in fields(PolicyBatch):
        values = [getattr(sample, field.name) for sample in samples]
        if values[0] is None:
            continue
        original_dtype = str(values[0].dtype).removeprefix("torch.")
        original = torch.stack(values)
        stored = storage_array(original)
        restored = torch.from_numpy(stored).to(original.dtype)
        if not torch.equal(original, restored):
            raise AssertionError(f"lossy packed conversion detected: {field.name}")
        arrays[field.name] = stored
        arrays[f"__dtype__{field.name}"] = np.asarray(original_dtype)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, output)
    return output, counts, output.stat().st_size, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Losslessly precompute MPCT PolicyBatch features")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ratio = config.data.goal_wait_keep_ratio
    manifest = args.manifest.resolve()
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.limit_episodes is not None:
        records = records[:args.limit_episodes]
    for record in records:
        missing = {"path", "arrival_steps", "time_steps"} - set(record)
        if missing:
            raise ValueError(f"manifest record is missing {sorted(missing)}: {record}")
        path = Path(record["path"])
        if not path.is_absolute():
            record["path"] = str((manifest.parent / path).resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dict = config.to_dict()["model"]
    data_dict = config.to_dict()["data"]
    tasks = [(r, str(output), ratio, args.overwrite, model_dict, data_dict) for r in records]
    converted, total, skipped = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (record, result) in enumerate(zip(records, pool.map(convert, tasks, chunksize=1)), 1):
            path, counts, size, was_skipped = result
            converted.append({
                "path": os.path.relpath(path, output),
                "samples": int(counts.sum()),
                "sample_counts": counts.tolist(),
                "map_family": record.get("map_family"),
                "num_agents": int(record.get("num_agents", len(counts))),
                "packed_policy_version": PACKED_POLICY_VERSION,
                "model_fingerprint": model_fingerprint(config.model),
                "exact_roundtrip_verified": True,
            })
            total += size
            skipped += int(was_skipped)
            if index == 1 or index % 100 == 0 or index == len(records):
                print(f"packed={index}/{len(records)} skipped={skipped} size_gib={total/1024**3:.3f}", flush=True)
    target = output / "manifest.jsonl"
    target.write_text("".join(json.dumps(record) + "\n" for record in converted), encoding="utf-8")
    metadata = {
        "source_manifest": str(manifest), "episodes": len(converted),
        "samples": sum(record["samples"] for record in converted), "bytes": total,
        "goal_wait_keep_ratio": ratio, "model_fingerprint": model_fingerprint(config.model),
        "packed_policy_version": PACKED_POLICY_VERSION,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
