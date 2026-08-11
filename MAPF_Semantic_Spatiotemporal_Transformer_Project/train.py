from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mapf_sst.checkpoint import save_checkpoint
from mapf_sst.config import ProjectConfig, load_config
from mapf_sst.data import EpisodeFeatureBuilder, EpisodeSequenceViewDataset, SyntheticPolicyDataset
from mapf_sst.losses import compute_loss
from mapf_sst.model import SemanticSpatiotemporalPolicy
from mapf_sst.types import PolicyBatch, stack_policy_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the revised MAPF semantic-token policy")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets(config: ProjectConfig):
    if config.data.kind == "synthetic":
        train = SyntheticPolicyDataset(config.model, config.data.train_samples, config.training.seed)
        val = SyntheticPolicyDataset(config.model, config.data.val_samples, config.training.seed + 100_000)
        return train, val
    if config.data.kind == "npz_manifest":
        if not config.data.train_manifest or not config.data.val_manifest:
            raise ValueError("npz_manifest data requires train_manifest and val_manifest")
        builder = EpisodeFeatureBuilder(
            config.model, coordinate_order=config.data.coordinate_order
        )
        train = EpisodeSequenceViewDataset(
            config.data.train_manifest, builder,
            goal_wait_keep_ratio=config.data.goal_wait_keep_ratio,
            max_samples=config.data.max_train_samples,
        )
        val_full = EpisodeSequenceViewDataset(
            config.data.val_manifest, builder,
            goal_wait_keep_ratio=config.data.goal_wait_keep_ratio,
        )
        val = balanced_validation_subset(val_full, config.data.max_val_samples)
        return train, val
    raise ValueError(f"unsupported data kind: {config.data.kind}")


def balanced_validation_subset(dataset: EpisodeSequenceViewDataset, limit: int | None):
    if limit is None or len(dataset) <= limit:
        return dataset
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    previous = 0
    for record, end_value in zip(dataset.records, dataset.cumulative):
        end = int(end_value)
        groups[(record["map_family"], record["num_agents"])].extend(range(previous, end))
        previous = end
    expected = [(family, agents) for family in ("maze", "random") for agents in (16, 24, 32)]
    missing = [key for key in expected if key not in groups]
    if missing:
        raise ValueError(f"balanced validation is missing strata: {missing}")
    selected: list[int] = []
    family_quota = int(limit) // 2
    for family in ("maze", "random"):
        base, remainder = divmod(family_quota, 3)
        for position, agents in enumerate((16, 24, 32)):
            quota = base + int(position < remainder)
            pool = groups[(family, agents)]
            offsets = np.linspace(0, len(pool) - 1, num=quota, dtype=np.int64)
            selected.extend(pool[int(offset)] for offset in offsets)
    selected.sort()
    return Subset(dataset, selected)


def evaluate(
    model: SemanticSpatiotemporalPolicy,
    loader: DataLoader,
    config: ProjectConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "correct": 0, "count": 0}
    with torch.no_grad():
        for batch in loader:
            assert isinstance(batch, PolicyBatch)
            batch = batch.to(device)
            output = model(batch, coordination_mode="none", return_reconstruction=True)
            loss = compute_loss(output, batch, config.model, config.training)
            totals["loss"] += float(loss.total) * batch.batch_size
            totals["correct"] += int((output.ego_logits.argmax(-1) == batch.ego_action).sum())
            totals["count"] += batch.batch_size
    count = max(totals["count"], 1)
    return {"loss": totals["loss"] / count, "ego_accuracy": totals["correct"] / count}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.training.communication_rounds:
        raise ValueError("train.py is the no-communication token baseline; use train_grouped.py")
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    seed_everything(config.training.seed)
    device = torch.device(args.device)
    train_set, val_set = make_datasets(config)
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=stack_policy_batches,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=stack_policy_batches,
    )

    model = SemanticSpatiotemporalPolicy(config.model).to(device)
    if config.model.freeze_map_encoder and not model.map_encoder.is_frozen:
        raise RuntimeError("freeze_map_encoder=true, but trainable map parameters remain")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    best = float("inf")
    global_step = 0

    for epoch in range(config.training.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(batch, coordination_mode="none", return_reconstruction=True)
                loss = compute_loss(output, batch, config.model, config.training)
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if config.training.max_steps is not None and global_step >= config.training.max_steps:
                break

        metrics = evaluate(model, val_loader, config, device)
        record = {"epoch": epoch, "step": global_step, **metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        if metrics["loss"] < best:
            best = metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                step=global_step,
                metrics=metrics,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            step=global_step,
            metrics=metrics,
        )
        if config.training.max_steps is not None and global_step >= config.training.max_steps:
            break


if __name__ == "__main__":
    main()
