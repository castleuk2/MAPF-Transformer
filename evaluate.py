from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mapf_map_transformer.config import experiment_config_from_dict
from mapf_map_transformer.data import build_datasets
from mapf_map_transformer.losses import MapReconstructionLoss
from mapf_map_transformer.model import StructuredMapTransformer
from mapf_map_transformer.trainer import evaluate_model
from mapf_map_transformer.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 25 structured map tokens with reconstruction/topology metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default=None, help="Metrics JSON path.")
    parser.add_argument("--visualization", default=None, help="Optional reconstruction PNG path.")
    parser.add_argument("--no-reachability", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = experiment_config_from_dict(checkpoint["config"])
    device = resolve_device(args.device)
    _, val_dataset = build_datasets(config.dataset)
    batch_size = args.batch_size or config.training.val_batch_size
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = StructuredMapTransformer(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    criterion = MapReconstructionLoss(config.loss, config.model)
    metrics = evaluate_model(
        model,
        loader,
        criterion,
        device,
        include_reachability=not args.no_reachability,
        visualization_path=Path(args.visualization) if args.visualization else None,
        visualization_count=config.training.save_visualizations,
    )
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    output_path = Path(args.output) if args.output else checkpoint_path.with_name("evaluation_metrics.json")
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"metrics={output_path}")


if __name__ == "__main__":
    main()
