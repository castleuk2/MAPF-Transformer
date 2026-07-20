from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from .checkpoint import load_model_from_checkpoint
from .dataset import SequenceSampleBuilder, load_episode
from .training import choose_device

ACTION_NAMES = ["WAIT", "UP", "DOWN", "LEFT", "RIGHT"]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run offline inference on a saved episode")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True, help="Episode .npz file")
    parser.add_argument("--ego", type=int, default=0)
    parser.add_argument("--time-step", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    model, payload = load_model_from_checkpoint(args.checkpoint, device=device)
    episode = load_episode(args.episode)
    time_step = args.time_step if args.time_step >= 0 else episode.time_steps - 1
    sample = SequenceSampleBuilder(model.config).build(episode, args.ego, time_step)
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if key not in {"ego_id", "time_step"}
    }
    with torch.no_grad():
        output = model(batch)
        probabilities = torch.softmax(output.logits / max(args.temperature, 1e-6), dim=-1)
        if args.sample:
            action = int(torch.multinomial(probabilities, 1).item())
        else:
            action = int(probabilities.argmax(dim=-1).item())
    result = {
        "episode": str(Path(args.episode)),
        "ego": args.ego,
        "time_step": time_step,
        "action": action,
        "action_name": ACTION_NAMES[action],
        "probabilities": probabilities.squeeze(0).cpu().tolist(),
        "checkpoint_step": int(payload.get("step", 0)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
