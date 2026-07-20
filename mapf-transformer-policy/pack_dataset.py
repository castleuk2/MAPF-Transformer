from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from mapf_transformer.config import ModelConfig, load_experiment_config
from mapf_transformer.dataset import load_episode, read_manifest
from mapf_transformer.packed_dataset import (
    build_packed_episode,
    packed_manifest_record,
    write_packed_manifest,
)


def output_path_for(source_path: str, output_dir: Path) -> Path:
    source = Path(source_path)
    digest = hashlib.blake2b(str(source.resolve()).encode("utf-8"), digest_size=8).hexdigest()
    return output_dir / "episodes" / f"{digest}_{source.name}"


def convert_one(
    task: tuple[dict[str, Any], str, ModelConfig, bool, bool]
) -> tuple[dict[str, Any], str, int, bool]:
    record, output_dir_raw, config, compress, overwrite = task
    output_dir = Path(output_dir_raw)
    output_path = output_path_for(record["path"], output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return record, str(output_path), output_path.stat().st_size, True

    arrays = build_packed_episode(load_episode(record["path"]), config)
    temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    saver = np.savez_compressed if compress else np.savez
    with temporary.open("wb") as stream:
        saver(stream, **arrays)
    os.replace(temporary, output_path)
    return record, str(output_path), output_path.stat().st_size, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute lossless bitmap/bitfield frame caches for MAPF Transformer"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--compress", action="store_true", help="Use ZIP compression (smaller, slower to load)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    experiment = load_experiment_config(args.config)
    config = experiment.model
    source_records = read_manifest(args.manifest.resolve())
    if args.limit_episodes is not None:
        source_records = source_records[: args.limit_episodes]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "manifest.jsonl"

    tasks = [
        (record, str(output_dir), config, bool(args.compress), bool(args.overwrite))
        for record in source_records
    ]
    converted: list[dict[str, Any]] = []
    total_bytes = 0
    skipped = 0
    results = map(convert_one, tasks)
    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(convert_one, tasks, chunksize=1)
    try:
        for index, (source, output_path_raw, size, was_skipped) in enumerate(results, 1):
            output_path = Path(output_path_raw)
            converted.append(
                packed_manifest_record(output_path, output_manifest, source, config)
            )
            total_bytes += int(size)
            skipped += int(was_skipped)
            if index == 1 or index % 100 == 0 or index == len(tasks):
                print(
                    f"packed={index}/{len(tasks)} skipped={skipped} "
                    f"size_gib={total_bytes / 1024**3:.3f}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    write_packed_manifest(converted, output_manifest)
    print(f"manifest={output_manifest}")
    print(f"episodes={len(converted)} bytes={total_bytes} skipped={skipped}")


if __name__ == "__main__":
    main()
