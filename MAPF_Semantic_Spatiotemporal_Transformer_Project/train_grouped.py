from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from mapf_sst.checkpoint import save_checkpoint
from mapf_sst.communication import MultiRoundCommunicationPolicy
from mapf_sst.config import load_config
from mapf_sst.data.npz_dataset import EpisodeFeatureBuilder, _read_manifest
from mapf_sst.data.synthetic import make_synthetic_communication_group
from mapf_sst.losses import compute_loss
from mapf_sst.model import SemanticSpatiotemporalPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train optional multi-round MAPF communication")
    parser.add_argument("--config", default="configs/communication.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rounds = config.training.communication_rounds
    if rounds < 1:
        raise ValueError("communication config must set communication_rounds >= 1")
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    device = torch.device(args.device)

    base = SemanticSpatiotemporalPolicy(config.model).to(device)
    model = MultiRoundCommunicationPolicy(base).to(device)
    if config.model.freeze_map_encoder and not model.local_policy.map_encoder.is_frozen:
        raise RuntimeError("freeze_map_encoder=true, but trainable map parameters remain")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = None
    builder = None
    if config.data.kind == "npz_manifest":
        if not config.data.train_manifest:
            raise ValueError("grouped NPZ training requires train_manifest")
        episode_paths = _read_manifest(config.data.train_manifest)
        if config.data.feature_backend == "cpp":
            from mapf_sst.cpp import CppEpisodeFeatureBuilder
            builder = CppEpisodeFeatureBuilder(config.model, coordinate_order=config.data.coordinate_order)
        else:
            builder = EpisodeFeatureBuilder(config.model, coordinate_order=config.data.coordinate_order)

    for step in range(args.steps):
        if config.data.kind == "synthetic":
            batch, graph = make_synthetic_communication_group(
                config.model, views=6, seed=config.training.seed + step
            )
        else:
            assert episode_paths is not None and builder is not None
            path = episode_paths[step % len(episode_paths)]
            episode = builder.load_episode(path)
            time_step = step % episode.actions.shape[0]
            batch, graph = builder.build_all_views(episode, time_step)
        batch, graph = batch.to(device), graph.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, graph, rounds=rounds)
        loss = compute_loss(output.final, batch, config.model, config.training)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, config.training.grad_clip_norm)
        optimizer.step()
        if step % 10 == 0 or step + 1 == args.steps:
            record = {"step": step + 1, "rounds": rounds, **loss.detached()}
            print(json.dumps(record, ensure_ascii=False))

    save_checkpoint(
        output_dir / "last.pt",
        model=base,
        optimizer=optimizer,
        config=config,
        epoch=0,
        step=args.steps,
        metrics={"communication_rounds": float(rounds)},
    )


if __name__ == "__main__":
    main()
