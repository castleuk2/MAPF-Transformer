from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mapf_map_transformer.config import experiment_config_from_dict
from mapf_map_transformer.model import StructuredMapTransformer
from mapf_map_transformer.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 25xD map tokens and reconstruction for a 17x17 input.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".npy or .npz with a 17x17 map (key: halo_map/halo_maps/maps).")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_input(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        key = next((candidate for candidate in ("halo_map", "halo_maps", "maps") if candidate in archive), None)
        if key is None:
            raise KeyError("NPZ must contain halo_map, halo_maps or maps.")
        array = archive[key]
    else:
        raise ValueError("Input must be .npy or .npz.")
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-2:] != (17, 17):
        raise ValueError(f"Expected [N,17,17], got {array.shape}")
    return np.asarray(array, dtype=np.int64)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = experiment_config_from_dict(checkpoint["config"])
    device = resolve_device(args.device)
    model = StructuredMapTransformer(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    inputs = load_input(Path(args.input))
    with torch.inference_mode():
        output = model(torch.from_numpy(inputs).to(device=device, dtype=torch.long), return_attention=True)
    attention = None
    if output.attention_maps:
        attention = torch.stack(output.attention_maps, dim=1).cpu().numpy()
    probabilities = (output.reconstruction_logits.softmax(dim=-1)[..., 1]
                     if output.reconstruction_logits.ndim == 4 else torch.sigmoid(output.reconstruction_logits))
    np.savez_compressed(
        args.output,
        halo_maps=inputs,
        latent_tokens=output.latent_tokens.cpu().numpy(),
        reconstruction_probabilities=probabilities.cpu().numpy(),
        attention_maps=attention if attention is not None else np.empty((0,), dtype=np.float32),
    )
    print(f"saved={args.output} tokens={tuple(output.latent_tokens.shape)}")


if __name__ == "__main__":
    main()
