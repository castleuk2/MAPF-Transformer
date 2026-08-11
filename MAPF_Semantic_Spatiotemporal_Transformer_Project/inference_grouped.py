from __future__ import annotations

import argparse
import json

import torch

from mapf_sst.checkpoint import load_checkpoint
from mapf_sst.communication import MultiRoundCommunicationPolicy
from mapf_sst.config import load_config
from mapf_sst.data.npz_dataset import EpisodeFeatureBuilder
from mapf_sst.data.synthetic import make_synthetic_communication_group
from mapf_sst.model import SemanticSpatiotemporalPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional learned message exchange among ego views")
    parser.add_argument("--config", default="configs/communication.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--npz")
    parser.add_argument("--time-step", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rounds = args.rounds or max(config.training.communication_rounds, 1)
    device = torch.device(args.device)
    base = SemanticSpatiotemporalPolicy(config.model).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=base, map_location=device)
    model = MultiRoundCommunicationPolicy(base).to(device).eval()
    if args.npz:
        builder = EpisodeFeatureBuilder(config.model, coordinate_order=config.data.coordinate_order)
        episode = builder.load_episode(args.npz)
        batch, graph = builder.build_all_views(episode, args.time_step)
    else:
        batch, graph = make_synthetic_communication_group(config.model, views=6, seed=7)
    batch, graph = batch.to(device), graph.to(device)
    with torch.no_grad():
        output = model(batch, graph, rounds=rounds)
    probabilities = torch.softmax(output.final.ego_logits, dim=-1).cpu()
    print(json.dumps({
        "communication_rounds": rounds,
        "action_order": ["WAIT", "UP", "DOWN", "LEFT", "RIGHT"],
        "preferences": probabilities.tolist(),
        "argmax_actions": probabilities.argmax(dim=-1).tolist(),
        "note": "Each row is the agent's own ego-view preference after learned messages.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
