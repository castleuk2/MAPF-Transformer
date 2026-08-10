from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from train import ROOT, build_dataset, evaluate, set_seed
from mapf_pct.checkpoint import save_checkpoint
from mapf_pct.config import load_config
from mapf_pct.losses import compute_loss
from mapf_pct.model import PreferenceCoordinationTransformer
from mapf_pct.types import stack_policy_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2-GPU DDP MPCT retraining with full loss tracking")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    config = load_config(args.config)
    if config.training.batch_size % world_size:
        raise ValueError("global batch_size must be divisible by world_size")
    local_batch = config.training.batch_size // world_size
    set_seed(config.training.seed + rank)
    output_dir = ROOT / args.output_dir
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    train_data = build_dataset(config, True)
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = DataLoader(
        train_data, batch_size=local_batch, sampler=sampler,
        num_workers=config.training.num_workers, collate_fn=stack_policy_batches,
        pin_memory=True, persistent_workers=config.training.num_workers > 0,
    )
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            build_dataset(config, False), batch_size=config.training.val_batch_size,
            shuffle=False, num_workers=config.training.num_workers,
            collate_fn=stack_policy_batches, pin_memory=True,
            persistent_workers=config.training.num_workers > 0,
        )

    raw_model = PreferenceCoordinationTransformer(config.model).to(device)
    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate, weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
    )
    total_updates = len(train_loader) * config.training.epochs
    def lr_factor(update: int) -> float:
        warmup = min(config.training.warmup_steps, total_updates)
        if warmup and update < warmup:
            return max(1, update) / warmup
        progress = (update - warmup) / max(1, total_updates - warmup)
        minimum = config.training.min_learning_rate / config.training.learning_rate
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp)
    best = float("inf"); step = 0
    component_names = ("action", "ego_action", "neighbor_action", "act_request", "map", "conflict", "reason", "scene_risk", "total")

    for epoch in range(config.training.epochs):
        sampler.set_epoch(epoch); model.train()
        sums = torch.zeros(len(component_names) + 2, device=device, dtype=torch.float64)
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config.training.amp):
                output = model(batch, return_map_reconstruction=True)
                losses = compute_loss(output, batch, config.model, config.training)
            scaler.scale(losses.total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer); scaler.update(); scheduler.step(); step += 1
            count = batch.batch_size
            for index, name in enumerate(component_names):
                value = losses.components.get(name)
                if value is not None: sums[index] += value.detach().double() * count
            sums[-2] += (output.ego_logits.argmax(-1) == batch.all_agent_actions[:, 0]).sum()
            sums[-1] += count
            if rank == 0 and step % 100 == 0:
                print(f"epoch={epoch+1} step={step} lr={optimizer.param_groups[0]['lr']:.8f}", flush=True)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        train_count = float(sums[-1])
        train_metrics = {f"train_{name}": float(sums[i] / train_count) for i, name in enumerate(component_names)}
        train_metrics["train_ego_accuracy"] = float(sums[-2] / sums[-1])

        if rank == 0:
            assert val_loader is not None
            val_metrics = evaluate(raw_model, val_loader, config, device)
            event = {"epoch": epoch + 1, "step": step, **train_metrics, **val_metrics}
            print(json.dumps(event, ensure_ascii=False), flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            save_checkpoint(output_dir / "last.pt", raw_model, project_config=config,
                            optimizer=optimizer, step=step, epoch=epoch+1, metrics=event)
            if event["val_total"] < best:
                best = event["val_total"]
                save_checkpoint(output_dir / "best.pt", raw_model, project_config=config,
                                optimizer=optimizer, step=step, epoch=epoch+1, metrics=event)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
