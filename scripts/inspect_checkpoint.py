from __future__ import annotations

import argparse
import json

import torch

from mapf_map_transformer.config import experiment_config_from_dict
from mapf_map_transformer.model import StructuredMapTransformer
from mapf_map_transformer.utils import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = experiment_config_from_dict(checkpoint["config"])
    model = StructuredMapTransformer(config.model)
    model.load_state_dict(checkpoint["model_state"])
    total, trainable = count_parameters(model)
    print(json.dumps({
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_metric": checkpoint.get("best_metric"),
        "tokens": model.num_latent_tokens,
        "d_model": config.model.d_model,
        "parameters": total,
        "trainable_parameters": trainable,
    }, indent=2))


if __name__ == "__main__":
    main()
