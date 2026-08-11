from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from train import evaluate, make_datasets, seed_everything
from mapf_sst.checkpoint import save_checkpoint
from mapf_sst.config import load_config
from mapf_sst.losses import compute_loss
from mapf_sst.model import SemanticSpatiotemporalPolicy
from mapf_sst.types import stack_policy_batches


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-GPU SST 3/6-epoch training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    config = load_config(args.config)
    if config.training.communication_rounds:
        raise ValueError("this run is the no-communication semantic-token baseline")
    # Match the earlier packed/raw trainers: batch_size is the micro-batch on
    # each GPU. With two ranks and no accumulation, 128 means effective 256.
    local_batch = config.training.batch_size
    seed_everything(config.training.seed + rank)
    output_dir = (ROOT / config.training.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    train_data, val_data = make_datasets(config)
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    train_loader = DataLoader(
        train_data,
        batch_size=local_batch,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        collate_fn=stack_policy_batches,
        pin_memory=True,
        persistent_workers=config.training.num_workers > 0,
    )
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_data,
            batch_size=config.training.val_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=stack_policy_batches,
            pin_memory=True,
            persistent_workers=config.training.num_workers > 0,
        )

    raw_model = SemanticSpatiotemporalPolicy(config.model).to(device)
    if config.model.freeze_map_encoder and not raw_model.map_encoder.is_frozen:
        raise RuntimeError("frozen Map contract failed")
    # The no-communication baseline intentionally leaves message-slot parameters
    # unused; DDP must therefore discover unused parameters for this run.
    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )
    total_updates = len(train_loader) * config.training.epochs
    warmup = min(2000, total_updates)

    def lr_factor(update: int) -> float:
        if warmup and update < warmup:
            return max(1, update) / warmup
        progress = (update - warmup) / max(1, total_updates - warmup)
        minimum = 0.1
        return minimum + 0.5 * (1.0 - minimum) * (
            1.0 + math.cos(math.pi * min(1.0, progress))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = bool(config.training.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    component_names = (
        "total", "ego_action", "ranking", "map_reconstruction", "semantic_reconstruction"
    )
    best = float("inf")
    step = 0
    log_time = time.perf_counter()

    for epoch in range(config.training.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        sums = torch.zeros(len(component_names) + 2, device=device, dtype=torch.float64)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp_enabled):
                output = model(batch, coordination_mode="none", return_reconstruction=False)
                losses = compute_loss(output, batch, config.model, config.training)
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            values = {
                "total": losses.total,
                "ego_action": losses.ego_action,
                "ranking": losses.ranking,
                "map_reconstruction": losses.map_reconstruction,
                "semantic_reconstruction": losses.semantic_reconstruction,
            }
            count = batch.batch_size
            for index, name in enumerate(component_names):
                sums[index] += values[name].detach().double() * count
            sums[-2] += (output.ego_logits.argmax(-1) == batch.ego_action).sum()
            sums[-1] += count
            if rank == 0 and step % 100 == 0:
                now = time.perf_counter()
                throughput = 100 * local_batch * world_size / (now - log_time)
                print(
                    f"epoch={epoch + 1} step={step} lr={optimizer.param_groups[0]['lr']:.8f} "
                    f"loss={float(losses.total.detach()):.6f} samples_per_s={throughput:.1f}",
                    flush=True,
                )
                log_time = now
            if args.max_updates is not None and step >= args.max_updates:
                break

        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        count = float(sums[-1])
        train_metrics = {
            f"train_{name}": float(sums[index] / count)
            for index, name in enumerate(component_names)
        }
        train_metrics["train_ego_accuracy"] = float(sums[-2] / sums[-1])

        if args.max_updates is not None and step >= args.max_updates:
            break

        if rank == 0:
            assert val_loader is not None
            val = evaluate(raw_model, val_loader, config, device)
            event = {
                "epoch": epoch + 1,
                "step": step,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                **train_metrics,
                "val_total": val["loss"],
                "val_ego_accuracy": val["ego_accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            print(json.dumps(event, ensure_ascii=False), flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            epoch_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                epoch_path, model=raw_model, optimizer=optimizer, config=config,
                epoch=epoch + 1, step=step, metrics=event,
            )
            shutil.copy2(epoch_path, output_dir / "last.pt")
            if event["val_total"] < best:
                best = event["val_total"]
                shutil.copy2(epoch_path, output_dir / "best.pt")
            if epoch + 1 in {3, 6}:
                shutil.copy2(epoch_path, output_dir / f"last_epoch{epoch + 1}.pt")
                shutil.copy2(output_dir / "best.pt", output_dir / f"best_epoch{epoch + 1}.pt")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
