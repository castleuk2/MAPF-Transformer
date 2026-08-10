from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mapf_pct.checkpoint import load_checkpoint
from mapf_pct.config import load_config
from mapf_pct.types import stack_policy_batches
from train import build_dataset, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.data.max_train_samples = args.samples
    config.data.max_val_samples = args.samples
    device = torch.device(args.device)
    model, payload = load_checkpoint(args.checkpoint, map_location="cpu")
    model = model.to(device).eval()
    result = {"checkpoint_epoch": payload.get("epoch"), "checkpoint_step": payload.get("step")}
    for split, train in (("train", True), ("val", False)):
        loader = DataLoader(
            build_dataset(config, train),
            batch_size=config.training.val_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=stack_policy_batches,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate(model, loader, config, device)
        result.update({key.replace("val_", f"{split}_", 1): value for key, value in metrics.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
