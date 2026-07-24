from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mapf_transformer.features import visible_neighbor_ids
from mapf_transformer.tracking import StableNeighborTracker


def precompute_tracking(
    positions: np.ndarray,
    max_neighbors: int = 24,
    radius: int = 7,
    grace_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, num_agents, _ = positions.shape
    ids = np.full((frames, num_agents, max_neighbors), -1, dtype=np.int16)
    valid = np.zeros((frames, num_agents, max_neighbors), dtype=bool)
    reset = np.zeros((frames, num_agents, max_neighbors), dtype=bool)
    trackers = [
        StableNeighborTracker(max_neighbors=max_neighbors, grace_steps=grace_steps)
        for _ in range(num_agents)
    ]
    for frame in range(frames):
        for ego_id in range(num_agents):
            visible, scores = visible_neighbor_ids(positions[frame], ego_id, radius)
            result = trackers[ego_id].update(visible, scores)
            ids[frame, ego_id] = result.agent_ids
            valid[frame, ego_id] = result.valid
            reset[frame, ego_id] = result.reset
    return ids, valid, reset


def upgrade_episode(path_text: str, overwrite: bool) -> tuple[str, str]:
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as source:
        if "neighbor_ids_24" in source and not overwrite:
            return str(path), "skipped"
        arrays = {key: np.asarray(source[key]) for key in source.files}
    ids, valid, reset = precompute_tracking(np.asarray(arrays["positions"], dtype=np.int16))
    arrays["neighbor_ids_24"] = ids
    arrays["neighbor_valid_24"] = valid
    arrays["track_reset_24"] = reset

    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.tracking24.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        # Validate the newly written archive before replacing the source.
        with np.load(temporary_path, allow_pickle=False) as check:
            if check["neighbor_ids_24"].shape[-1] != 24:
                raise RuntimeError("tracking-24 validation failed")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(path), "updated"


def manifest_paths(manifest: Path) -> list[Path]:
    paths = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            path = Path(record["path"])
            if not path.is_absolute():
                path = manifest.parent / path
            paths.append(path.resolve())
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically add full-episode 24-slot tracking arrays to existing NPZ episodes."
    )
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    unique_paths = sorted({
        path
        for manifest_text in args.manifest
        for path in manifest_paths(Path(manifest_text))
    })
    updated = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(upgrade_episode, str(path), args.overwrite): path
            for path in unique_paths
        }
        for index, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                _, status = future.result()
                updated += status == "updated"
                skipped += status == "skipped"
            except Exception as error:
                failed += 1
                print(f"FAILED {path}: {error}", flush=True)
            if index % 100 == 0 or index == len(futures):
                print(
                    f"processed={index}/{len(futures)} updated={updated} "
                    f"skipped={skipped} failed={failed}",
                    flush=True,
                )
    if failed:
        raise SystemExit(f"{failed} episodes failed")


if __name__ == "__main__":
    main()
