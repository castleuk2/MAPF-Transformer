from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .checkpoint import load_checkpoint_payload, save_checkpoint
from .config import ExperimentConfig, load_experiment_config, save_experiment_config
from .dataset import EpisodeSequenceDataset
from .model import MAPFTransformer


def append_metric(metrics_file: Path, event: str, **values: object) -> None:
    record = {"event": event, "time": time.time(), **values}
    with metrics_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def distributed_context(requested_device: str) -> tuple[torch.device, int, int, int, bool]:
    """Initializes torchrun DDP and returns device/rank metadata."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training currently requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", local_rank), rank, local_rank, world_size, True
    return choose_device(requested_device), rank, local_rank, world_size, False


def local_accumulation_steps(global_steps: int, world_size: int) -> int:
    if global_steps <= 0 or world_size <= 0:
        raise ValueError("accumulation steps and world size must be positive")
    if global_steps % world_size != 0:
        raise ValueError(
            "gradient_accumulation_steps must be divisible by the number of DDP processes"
        )
    return global_steps // world_size


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.no_grad()
def evaluate(
    model: MAPFTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_reconstruction_loss = 0.0
    total_correct = 0
    total_examples = 0
    batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        # Compute the auxiliary objective during validation as well so train and
        # validation metrics use the same loss definition. Inference keeps the
        # reconstruction decoder disabled by default while the model is in eval mode.
        output = model(batch, return_reconstruction=True)
        count = int(batch["target"].numel())
        if output.loss is not None:
            total_loss += float(output.loss) * count
        if output.action_loss is not None:
            total_action_loss += float(output.action_loss) * count
        if output.map_reconstruction_loss is not None:
            total_reconstruction_loss += float(output.map_reconstruction_loss) * count
        total_correct += int((output.logits.argmax(dim=-1) == batch["target"]).sum())
        total_examples += count
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    model.train()
    divisor = max(1, total_examples)
    return {
        "loss": total_loss / divisor,
        "action_loss": total_action_loss / divisor,
        "map_reconstruction_loss": total_reconstruction_loss / divisor,
        "accuracy": total_correct / divisor,
        "examples": float(total_examples),
    }


def train(config: ExperimentConfig, resume: str | None = None) -> Path:
    config.validate()
    train_cfg = config.training
    model_cfg = config.model
    device, rank, local_rank, world_size, distributed = distributed_context(train_cfg.device)
    is_main = rank == 0
    set_seed(train_cfg.seed + rank)
    accumulation_per_rank = local_accumulation_steps(
        train_cfg.gradient_accumulation_steps, world_size
    )
    output_dir = Path(train_cfg.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_experiment_config(config, output_dir / "resolved_config.yaml")
        if resume is None:
            metrics_path.write_text("", encoding="utf-8")
    if distributed:
        dist.barrier()

    train_dataset = EpisodeSequenceDataset(
        train_cfg.train_manifest,
        model_cfg,
        history_augmentation=train_cfg.history_augmentation,
        min_history_frames=train_cfg.min_history_frames,
        goal_wait_keep_ratio=train_cfg.goal_wait_keep_ratio,
        max_samples=train_cfg.max_train_samples,
        seed=train_cfg.seed,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=train_cfg.seed
    ) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = None
    if train_cfg.val_manifest and Path(train_cfg.val_manifest).exists():
        val_dataset = EpisodeSequenceDataset(
            train_cfg.val_manifest,
            model_cfg,
            history_augmentation=False,
            goal_wait_keep_ratio=train_cfg.goal_wait_keep_ratio,
            max_samples=None,
            seed=train_cfg.seed + 1,
        )
        if train_cfg.max_val_samples is not None and len(val_dataset) > train_cfg.max_val_samples:
            # Cover the entire held-out corpus instead of taking only the first
            # manifest episodes/maps.
            val_indices = np.linspace(
                0,
                len(val_dataset) - 1,
                num=int(train_cfg.max_val_samples),
                dtype=np.int64,
            ).tolist()
            val_dataset = Subset(val_dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.val_batch_size or train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=device.type == "cuda",
        )

    raw_model = MAPFTransformer(model_cfg).to(device)
    model: MAPFTransformer | DDP = raw_model
    if distributed:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_per_rank))
    total_steps = train_cfg.max_steps or (train_cfg.epochs * steps_per_epoch)
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=train_cfg.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=train_cfg.min_learning_rate / train_cfg.learning_rate,
    )

    global_step = 0
    start_epoch = 0
    best_val = float("inf")
    if resume:
        payload = load_checkpoint_payload(resume, map_location=device)
        raw_model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        global_step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0))
        best_val = float(payload.get("metrics", {}).get("best_val_loss", best_val))

    use_amp = train_cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    running_loss = 0.0
    running_action_loss = 0.0
    running_reconstruction_loss = 0.0
    running_examples = 0
    last_log_time = time.time()

    if is_main:
        effective_batch = train_cfg.batch_size * train_cfg.gradient_accumulation_steps
        print(
            f"training device={device} world_size={world_size} micro_batch={train_cfg.batch_size} "
            f"global_accumulation={train_cfg.gradient_accumulation_steps} "
            f"effective_batch={effective_batch} steps_per_epoch={steps_per_epoch} "
            f"total_steps={total_steps}",
            flush=True,
        )
        append_metric(
            metrics_path,
            "start",
            device=str(device),
            world_size=world_size,
            model_parameters=sum(parameter.numel() for parameter in raw_model.parameters()),
            train_samples=len(train_dataset),
            val_samples=len(val_loader.dataset) if val_loader is not None else 0,
            micro_batch=train_cfg.batch_size,
            global_accumulation=train_cfg.gradient_accumulation_steps,
            effective_batch=effective_batch,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            goal_wait_keep_ratio=train_cfg.goal_wait_keep_ratio,
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, train_cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        loader_batches = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            if global_step >= total_steps:
                break
            group_offset = batch_index % accumulation_per_rank
            group_size = min(accumulation_per_rank, loader_batches - batch_index + group_offset)
            should_step = group_offset + 1 == group_size
            batch = move_batch(batch, device)
            sync_context = nullcontext() if (not distributed or should_step) else model.no_sync()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    output = model(batch)
                    if output.loss is None:
                        raise RuntimeError("Training batch did not produce a loss")
                    loss = output.loss / group_size
                scaler.scale(loss).backward()

            count = int(batch["target"].numel())
            running_loss += float(output.loss.detach()) * count
            if output.action_loss is not None:
                running_action_loss += float(output.action_loss.detach()) * count
            if output.map_reconstruction_loss is not None:
                running_reconstruction_loss += (
                    float(output.map_reconstruction_loss.detach()) * count
                )
            running_examples += count
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if is_main and global_step % train_cfg.log_every == 0:
                elapsed = max(1e-6, time.time() - last_log_time)
                average = running_loss / max(1, running_examples)
                average_action = running_action_loss / max(1, running_examples)
                average_reconstruction = running_reconstruction_loss / max(1, running_examples)
                throughput = running_examples * world_size / elapsed
                print(
                    f"step={global_step} epoch={epoch + 1} total_loss={average:.5f} "
                    f"action_loss={average_action:.5f} "
                    f"map_reconstruction_loss={average_reconstruction:.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} samples/s={throughput:.1f}",
                    flush=True,
                )
                append_metric(
                    metrics_path,
                    "train",
                    step=global_step,
                    epoch=epoch + 1,
                    loss=average,
                    total_loss=average,
                    action_loss=average_action,
                    map_reconstruction_loss=average_reconstruction,
                    weighted_map_reconstruction_loss=(
                        model_cfg.aux_map_loss_weight * average_reconstruction
                    ),
                    learning_rate=scheduler.get_last_lr()[0],
                    samples_per_second=throughput,
                )
                running_loss = 0.0
                running_action_loss = 0.0
                running_reconstruction_loss = 0.0
                running_examples = 0
                last_log_time = time.time()

            if is_main and val_loader is not None and global_step % train_cfg.validate_every == 0:
                metrics = evaluate(raw_model, val_loader, device)
                print(
                    f"validation step={global_step} total_loss={metrics['loss']:.5f} "
                    f"action_loss={metrics['action_loss']:.5f} "
                    f"map_reconstruction_loss={metrics['map_reconstruction_loss']:.5f} "
                    f"accuracy={metrics['accuracy']:.4f}",
                    flush=True,
                )
                append_metric(
                    metrics_path,
                    "validation",
                    step=global_step,
                    epoch=epoch + 1,
                    loss=metrics["loss"],
                    total_loss=metrics["loss"],
                    action_loss=metrics["action_loss"],
                    map_reconstruction_loss=metrics["map_reconstruction_loss"],
                    weighted_map_reconstruction_loss=(
                        model_cfg.aux_map_loss_weight * metrics["map_reconstruction_loss"]
                    ),
                    accuracy=metrics["accuracy"],
                    examples=metrics["examples"],
                )
                if metrics["loss"] < best_val:
                    best_val = metrics["loss"]
                    metrics["best_val_loss"] = best_val
                    save_checkpoint(
                        output_dir / "best.pt",
                        raw_model,
                        config,
                        optimizer,
                        scheduler,
                        step=global_step,
                        epoch=epoch,
                        metrics=metrics,
                    )

            if is_main and global_step % train_cfg.save_every == 0:
                save_checkpoint(
                    output_dir / f"step_{global_step:08d}.pt",
                    raw_model,
                    config,
                    optimizer,
                    scheduler,
                    step=global_step,
                    epoch=epoch,
                    metrics={"best_val_loss": best_val},
                )
                append_metric(
                    metrics_path,
                    "checkpoint",
                    step=global_step,
                    epoch=epoch + 1,
                    path=str(output_dir / f"step_{global_step:08d}.pt"),
                )
        if global_step >= total_steps:
            break

    final_path = output_dir / "last.pt"
    if is_main:
        final_metrics = {"best_val_loss": best_val}
        if val_loader is not None:
            final_metrics.update(evaluate(raw_model, val_loader, device))
        save_checkpoint(
            final_path,
            raw_model,
            config,
            optimizer,
            scheduler,
            step=global_step,
            epoch=train_cfg.epochs,
            metrics=final_metrics,
        )
        append_metric(
            metrics_path,
            "complete",
            step=global_step,
            epoch=train_cfg.epochs,
            path=str(final_path),
            validation_loss=final_metrics.get("loss"),
            validation_total_loss=final_metrics.get("loss"),
            validation_action_loss=final_metrics.get("action_loss"),
            validation_map_reconstruction_loss=final_metrics.get("map_reconstruction_loss"),
            validation_weighted_map_reconstruction_loss=(
                model_cfg.aux_map_loss_weight * final_metrics["map_reconstruction_loss"]
                if final_metrics.get("map_reconstruction_loss") is not None
                else None
            ),
            validation_accuracy=final_metrics.get("accuracy"),
            best_validation_loss=final_metrics.get("best_val_loss"),
        )
        print(f"Saved final checkpoint: {final_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return final_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MAPF Transformer policy")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    if args.train_manifest:
        config.training.train_manifest = args.train_manifest
    if args.val_manifest:
        config.training.val_manifest = args.val_manifest
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.device:
        config.training.device = args.device
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
