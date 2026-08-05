from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mapf_map_transformer.policy_data import crop_halo_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = args.output_dir / "episodes"
    episode_dir.mkdir(exist_ok=True)
    output_manifest = args.output_dir / "manifest.jsonl"
    count = 0
    with output_manifest.open("w", encoding="utf-8") as stream:
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            source = Path(record["source_path"])
            with np.load(source, allow_pickle=False) as archive:
                obstacles = np.asarray(archive["obstacles"], dtype=np.uint8)
                positions = np.asarray(archive["positions"], dtype=np.int16)
            frames = int(record["time_steps"])
            agents = int(record["num_agents"])
            packed = np.empty((frames, agents, 29), dtype=np.uint8)
            for frame in range(frames):
                for agent in range(agents):
                    core = crop_halo_map(obstacles, positions[frame, agent])[1:-1, 1:-1]
                    packed[frame, agent] = np.packbits(core.reshape(-1), bitorder="little")
            destination = episode_dir / f"episode_{count:07d}.npz"
            np.savez_compressed(destination, local_map_bits=packed)
            output = {
                **record,
                "path": str(destination.resolve()),
                "packed_format": {"version": 1, "map_size": 15},
            }
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"packed={count}", flush=True)
    print(f"episodes={count} manifest={output_manifest}")


if __name__ == "__main__":
    main()
