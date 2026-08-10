from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from mapf_pct.checkpoint import load_checkpoint
from mapf_pct.config import ModelConfig
from mapf_pct.constants import ACTION_NAMES, REASON_NAMES
from mapf_pct.data import EpisodeFeatureBuilder, make_synthetic_sample
from mapf_pct.model import PreferenceCoordinationTransformer
from mapf_pct.resolver import CSPIBTResolver
from mapf_pct.types import stack_policy_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for one ego-centered MAPF sample")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--ego", type=int, default=0)
    parser.add_argument("--time-step", type=int, default=0)
    parser.add_argument("--coordinate-order", choices=("row_col", "xy"), default="row_col")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, map_location=device)
        config = model.config
    else:
        config = ModelConfig(d_model=64, n_heads=4, coordination_layers=2, dropout=0.0)
        model = PreferenceCoordinationTransformer(config)
        payload = {"warning": "randomly initialized model"}
    model = model.to(device).eval()

    if args.episode:
        builder = EpisodeFeatureBuilder(config, coordinate_order=args.coordinate_order)
        sample = builder.build(args.episode, args.time_step, args.ego)
        source = str(args.episode)
    else:
        sample = make_synthetic_sample(config, args.seed)
        source = f"synthetic:{args.seed}"
    batch = stack_policy_batches([sample]).to(device)

    with torch.no_grad():
        output = model(batch, return_map_reconstruction=False)
        probabilities = output.all_agent_logits.softmax(dim=-1)[0]
        ego_probabilities = output.ego_logits.softmax(dim=-1)[0]

    valid = batch.agent_valid[0]
    # The local 15x15 resolver is a demonstration. Production integration should
    # pass global positions and the global obstacle map to the same resolver API.
    resolver = CSPIBTResolver()
    resolution = resolver.resolve(
        batch.agent_xy[0],
        output.all_agent_logits[0],
        batch.local_maps[0, 1:-1, 1:-1],
        valid=valid,
        on_goal=batch.on_goal[0],
    )
    agents = []
    for agent in range(config.max_agents):
        if not bool(valid[agent]):
            continue
        reason_names = []
        if output.reason_logits is not None:
            reason_prob = output.reason_logits[0, agent].sigmoid()
            reason_names = [REASON_NAMES[i] for i in torch.nonzero(reason_prob > 0.5).flatten().tolist()]
        agents.append(
            {
                "slot": agent,
                "position": batch.agent_xy[0, agent].tolist(),
                "preference": {
                    ACTION_NAMES[action]: round(float(probabilities[agent, action]), 6)
                    for action in range(config.num_actions)
                },
                "raw_argmax": ACTION_NAMES[int(probabilities[agent].argmax())],
                "resolved": ACTION_NAMES[int(resolution.actions[agent])],
                "reason_over_0.5": reason_names,
            }
        )
    result = {
        "source": source,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_metadata": {key: payload.get(key) for key in ("step", "epoch", "warning") if key in payload},
        "ego_preference": {
            ACTION_NAMES[action]: round(float(ego_probabilities[action]), 6)
            for action in range(config.num_actions)
        },
        "scene_risk": (
            output.scene_risk_logits.softmax(dim=-1)[0].tolist()
            if output.scene_risk_logits is not None
            else None
        ),
        "resolver_changes": int(resolution.changed_from_argmax.sum()),
        "agents": agents,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
