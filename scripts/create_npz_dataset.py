from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mapf_map_transformer.synthetic import generate_halo_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/validation .npz halo-map datasets.")
    parser.add_argument("--output-dir", default="data/halo_maps")
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_split(count: int, seed: int) -> np.ndarray:
    patterns = ("bernoulli", "walls", "rooms", "corridors", "mixed")
    maps = np.empty((count, 17, 17), dtype=np.uint8)
    for index in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        maps[index] = generate_halo_map(rng, pattern=str(rng.choice(patterns)))
    return maps


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train.npz", halo_maps=make_split(args.train_samples, args.seed))
    np.savez_compressed(output_dir / "val.npz", halo_maps=make_split(args.val_samples, args.seed + 1_000_003))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
