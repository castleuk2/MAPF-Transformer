from __future__ import annotations

import argparse
import json

import torch

from mapf_sst.checkpoint import load_checkpoint
from mapf_sst.config import load_config
from mapf_sst.data.npz_dataset import EpisodeFeatureBuilder
from mapf_sst.data.synthetic import make_synthetic_policy_batch
from mapf_sst.model import SemanticSpatiotemporalPolicy
from mapf_sst.types import stack_policy_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ego-centered MAPF preference inference")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--npz")
    parser.add_argument("--time-step", type=int, default=0)
    parser.add_argument("--ego", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    model = SemanticSpatiotemporalPolicy(config.model).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()
    if args.npz:
        builder = EpisodeFeatureBuilder(config.model, coordinate_order=config.data.coordinate_order)
        episode = builder.load_episode(args.npz)
        batch = stack_policy_batches([builder.build(episode, args.time_step, args.ego)])
    else:
        batch = make_synthetic_policy_batch(config.model, batch_size=1, seed=args.seed)
    batch = batch.to(device)
    with torch.no_grad():
        output = model(batch, coordination_mode="none")
    probabilities = torch.softmax(output.ego_logits, dim=-1)[0].cpu().tolist()
    print(json.dumps({
        "ego": args.ego,
        "action_order": ["WAIT", "UP", "DOWN", "LEFT", "RIGHT"],
        "probabilities": probabilities,
        "argmax": int(torch.tensor(probabilities).argmax()),
        "note": "Preference output only; no joint-action backend is assumed.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
