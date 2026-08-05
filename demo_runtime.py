from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mapf_map_transformer.actions import ACTION_TO_DELTA, Action
from mapf_map_transformer.config import ModelConfig, experiment_config_from_dict
from mapf_map_transformer.model import StructuredMapTransformer
from mapf_map_transformer.runtime import GlobalMapWindowProvider, MapTokenRuntime
from mapf_map_transformer.synthetic import generate_global_map
from mapf_map_transformer.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate 17-cell rolling updates and exact latent reuse.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def load_model(checkpoint_path: str | None, device: torch.device) -> StructuredMapTransformer:
    if checkpoint_path is None:
        model = StructuredMapTransformer(ModelConfig(dropout=0.0)).to(device)
        print("No checkpoint supplied: using randomly initialized weights for runtime mechanics only.")
        return model
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = experiment_config_from_dict(checkpoint["config"])
    model = StructuredMapTransformer(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    global_map = generate_global_map(args.seed, size=49)
    provider = GlobalMapWindowProvider(global_map, pad_value=1)
    center = (global_map.shape[0] // 2, global_map.shape[1] // 2)
    runtime = MapTokenRuntime(load_model(args.checkpoint, device), device=device)
    initial = runtime.reset(provider.crop(center))
    print(f"reset: version={initial.version} encode_count={runtime.encode_count} tokens={tuple(initial.latent_tokens.shape)}")

    actions = [Action.WAIT, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.WAIT, Action.LEFT, Action.UP]
    for step_index, action in enumerate(actions, start=1):
        old_center = center
        dr, dc = ACTION_TO_DELTA[action]
        candidate = (center[0] + dr, center[1] + dc)
        moved = action != Action.WAIT and global_map[candidate] == 0
        if moved:
            center = candidate
            strip = provider.incoming_strip(center, action)
            result = runtime.step(action, moved=True, incoming_strip=strip)
        else:
            result = runtime.step(action, moved=False)
        expected = provider.crop(center)
        if not np.array_equal(result.halo_map, expected):
            raise RuntimeError(f"Rolling update mismatch at step {step_index}: {old_center}->{center} action={action.name}")
        print(
            f"step={step_index} action={action.name:<5} moved={moved} reused={result.reused} "
            f"version={result.version} encodes={runtime.encode_count} reuses={runtime.reuse_count}"
        )
    print("Rolling 17-cell updates are exactly equivalent to fresh 17x17 crops.")


if __name__ == "__main__":
    main()
