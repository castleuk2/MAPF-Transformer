from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mapf_pct.checkpoint import save_checkpoint
from mapf_pct.config import ProjectConfig, load_config
from mapf_pct.data import EpisodeSequenceSampleDataset, SyntheticPolicyDataset
from mapf_pct.losses import compute_loss
from mapf_pct.model import PreferenceCoordinationTransformer
from mapf_pct.types import stack_policy_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MAPF Preference-Coordination Transformer")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "tiny.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config: ProjectConfig, train: bool):
    data = config.data
    seed = config.training.seed if train else config.training.seed + 1
    if data.kind == "synthetic":
        size = data.train_samples if train else data.val_samples
        return SyntheticPolicyDataset(config.model, size=size, seed=seed)
    if data.kind == "npz_manifest":
        manifest = data.train_manifest if train else data.val_manifest
        if manifest is None:
            raise ValueError("train_manifest/val_manifest is required for npz_manifest data")
        dataset = EpisodeSequenceSampleDataset(
            config.model,
            manifest,
            goal_wait_keep_ratio=data.goal_wait_keep_ratio,
            max_samples=data.max_train_samples if train else None,
            coordinate_order=data.coordinate_order,
            task_mode=data.task_mode,
            target_mode=data.target_mode,
        )
        if train or data.max_val_samples is None or len(dataset) <= data.max_val_samples:
            return dataset
        return balanced_validation_subset(dataset, int(data.max_val_samples))
    raise ValueError(f"unsupported data.kind: {data.kind}")


def balanced_validation_subset(dataset: EpisodeSequenceSampleDataset, sample_limit: int) -> Subset:
    """Equal Maze/Random split, then near-equal 16/24/32 strata.

    Indices are spread uniformly through every stratum so validation covers the
    complete held-out manifest instead of taking its first episodes.
    """
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    previous = 0
    for record, end_value in zip(dataset.records, dataset.cumulative):
        end = int(end_value)
        key = (str(record["map_family"]), int(record["num_agents"]))
        groups[key].extend(range(previous, end))
        previous = end
    expected = [(family, agents) for family in ("maze", "random") for agents in (16, 24, 32)]
    missing = [key for key in expected if key not in groups]
    if missing:
        raise ValueError(f"balanced validation is missing strata: {missing}")
    family_quota = sample_limit // 2
    selected: list[int] = []
    for family in ("maze", "random"):
        base, remainder = divmod(family_quota, 3)
        for position, agents in enumerate((16, 24, 32)):
            quota = base + int(position < remainder)
            pool = groups[(family, agents)]
            if len(pool) < quota:
                raise ValueError(f"validation stratum {(family, agents)} has {len(pool)} < {quota}")
            offsets = np.linspace(0, len(pool) - 1, num=quota, dtype=np.int64)
            selected.extend(pool[int(offset)] for offset in offsets)
    # Preserve episode locality for worker NPZ/distance-map cache reuse.
    selected.sort()
    return Subset(dataset, selected)


def evaluate(model, loader, config, device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    correct = 0
    examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, return_map_reconstruction=True)
            losses = compute_loss(output, batch, config.model, config.training)
            for key, value in losses.components.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            correct += int((output.ego_logits.argmax(dim=-1) == batch.all_agent_actions[:, 0]).sum())
            examples += batch.batch_size
            batches += 1
    result = {f"val_{key}": value / max(1, batches) for key, value in totals.items()}
    result["val_ego_accuracy"] = correct / max(1, examples)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    set_seed(config.training.seed)
    device = torch.device(args.device)
    output_dir = ROOT / config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    train_loader = DataLoader(
        build_dataset(config, True),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=stack_policy_batches,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        build_dataset(config, False),
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=stack_policy_batches,
        pin_memory=device.type == "cuda",
    )
    model = PreferenceCoordinationTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.training.epochs
    if config.training.max_steps is not None:
        total_updates = min(total_updates, config.training.max_steps)

    def lr_factor(update: int) -> float:
        warmup = min(config.training.warmup_steps, total_updates)
        if warmup and update < warmup:
            return max(1, update) / warmup
        progress = (update - warmup) / max(1, total_updates - warmup)
        minimum = config.training.min_learning_rate / config.training.learning_rate
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor) if config.training.use_scheduler else None
    use_amp = bool(config.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics_path = output_dir / "metrics.jsonl"
    best = float("inf")
    step = 0

    for epoch in range(config.training.epochs):
        model.train()
        train_totals: dict[str, float] = {}
        train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch, return_map_reconstruction=True)
                losses = compute_loss(output, batch, config.model, config.training)
                scaled_loss = losses.total / config.training.gradient_accumulation_steps
            for key, value in losses.components.items():
                train_totals[key] = train_totals.get(key, 0.0) + float(value.detach())
            train_batches += 1
            scaler.scale(scaled_loss).backward()
            update_now = (
                micro_step % config.training.gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )
            if update_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            if update_now and (step == 1 or step % 10 == 0):
                values = {key: round(float(value.detach()), 5) for key, value in losses.components.items()}
                values["lr"] = round(float(optimizer.param_groups[0]["lr"]), 8)
                print(f"epoch={epoch + 1} step={step} {values}")
            if update_now and config.training.max_steps is not None and step >= config.training.max_steps:
                break

        train_metrics = {
            f"train_{key}": value / max(1, train_batches)
            for key, value in train_totals.items()
        }
        val = evaluate(model, val_loader, config, device)
        event = {"epoch": epoch + 1, "step": step, **train_metrics, **val}
        print(json.dumps(event, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        save_checkpoint(
            output_dir / "last.pt", model, project_config=config,
            optimizer=optimizer, step=step, epoch=epoch + 1, metrics=val,
        )
        if val["val_total"] < best:
            best = val["val_total"]
            save_checkpoint(
                output_dir / "best.pt", model, project_config=config,
                optimizer=optimizer, step=step, epoch=epoch + 1, metrics=val,
            )
        if config.training.max_steps is not None and step >= config.training.max_steps:
            break

    print(f"saved checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
