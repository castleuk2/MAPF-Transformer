from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from mapf_pct.config import load_config
from mapf_pct.data import EpisodeSequenceSampleDataset, PackedPolicyDataset
from mapf_pct.types import PolicyBatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify packed MPCT features against source NPZ")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--packed-manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000, help="0 means exhaustive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    raw = EpisodeSequenceSampleDataset(
        config.model, args.source_manifest,
        goal_wait_keep_ratio=config.data.goal_wait_keep_ratio,
        coordinate_order=config.data.coordinate_order,
        task_mode=config.data.task_mode,
        target_mode=config.data.target_mode,
    )
    packed = PackedPolicyDataset(config.model, args.packed_manifest)
    if len(raw) != len(packed):
        raise AssertionError(f"length mismatch: raw={len(raw)} packed={len(packed)}")
    count = len(raw) if args.samples == 0 else min(len(raw), args.samples)
    indices = np.arange(len(raw)) if count == len(raw) else np.sort(
        np.random.default_rng(args.seed).choice(len(raw), count, replace=False)
    )
    checked_tensors = 0
    for ordinal, index in enumerate(indices, 1):
        source, restored = raw[int(index)], packed[int(index)]
        for field in fields(PolicyBatch):
            left, right = getattr(source, field.name), getattr(restored, field.name)
            if left is None or right is None:
                if left is not None or right is not None:
                    raise AssertionError(f"None mismatch index={index} field={field.name}")
                continue
            if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
                delta = 0.0 if left.numel() == 0 else float((left.float() - right.float()).abs().max())
                raise AssertionError(
                    f"mismatch index={index} field={field.name} dtype={left.dtype}/{right.dtype} "
                    f"shape={tuple(left.shape)}/{tuple(right.shape)} max_abs={delta}"
                )
            checked_tensors += 1
        if ordinal % 10000 == 0:
            print(f"verified={ordinal}/{count}", flush=True)
    print(json.dumps({
        "source_samples": len(raw), "packed_samples": len(packed),
        "checked_samples": count, "checked_tensors": checked_tensors,
        "exact_equal": True, "float_max_abs": 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
