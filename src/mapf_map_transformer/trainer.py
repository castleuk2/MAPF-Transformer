from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .config import ExperimentConfig, save_experiment_config
from .data import build_dataloaders
from .losses import MapReconstructionLoss
from .metrics import MetricAccumulator, binary_reconstruction_metrics
from .model import StructuredMapTransformer
from .utils import (
    append_jsonl,
    atomic_torch_save,
    cosine_with_warmup,
    count_parameters,
    resolve_device,
    set_global_seed,
)
from .visualization import save_reconstruction_grid


@dataclass(slots=True)
class TrainResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_metric: float
    global_step: int


def _checkpoint_payload(
    *,
    model: StructuredMapTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: ExperimentConfig,
    epoch: int,
    global_step: int,
    best_metric: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }


@torch.inference_mode()
def evaluate_model(
    model: StructuredMapTransformer,
    loader: DataLoader[Any],
    criterion: MapReconstructionLoss,
    device: torch.device,
    *,
    include_reachability: bool = True,
    visualization_path: Path | None = None,
    visualization_count: int = 8,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    visualized = False
    for batch in loader:
        halo_maps = batch["halo_map"].to(device=device, dtype=torch.long, non_blocking=True)
        output = model(halo_maps)
        loss = criterion(output, halo_maps)
        metrics = binary_reconstruction_metrics(
            output.reconstruction_logits,
            halo_maps,
            occupied_state=model.config.occupied_state,
            include_reachability=include_reachability,
        )
        metrics.update(loss.detached())
        accumulator.update(metrics, weight=float(halo_maps.shape[0]))
        if visualization_path is not None and not visualized:
            save_reconstruction_grid(
                halo_maps,
                output.reconstruction_logits,
                visualization_path,
                max_items=visualization_count,
            )
            visualized = True
    return accumulator.compute()


def train_experiment(config: ExperimentConfig) -> TrainResult:
    config.validate()
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "resolved_config.yaml")
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    train_loader, val_loader = build_dataloaders(config)
    model = StructuredMapTransformer(config.model).to(device)
    criterion = MapReconstructionLoss(config.loss, config.model)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    steps_per_epoch = max(len(train_loader), 1)
    planned_steps = config.training.epochs * steps_per_epoch
    if config.training.max_steps is not None:
        planned_steps = min(planned_steps, config.training.max_steps)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup(
            step,
            warmup_steps=config.training.warmup_steps,
            total_steps=max(planned_steps, 1),
        ),
    )

    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_parameters, trainable_parameters = count_parameters(model)
    print(
        f"device={device} parameters={total_parameters:,} trainable={trainable_parameters:,} "
        f"tokens={model.num_latent_tokens} d_model={config.model.d_model}"
    )

    global_step = 0
    best_metric = float("inf")
    best_checkpoint = output_dir / "best.pt"
    last_checkpoint = output_dir / "last.pt"
    stop_training = False

    for epoch in range(config.training.epochs):
        model.train()
        epoch_accumulator = MetricAccumulator()
        start_time = time.time()
        for batch_index, batch in enumerate(train_loader):
            halo_maps = batch["halo_map"].to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(halo_maps)
                loss = criterion(output, halo_maps)

            scaler.scale(loss.total).backward()
            if config.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            loss_values = loss.detached()
            epoch_accumulator.update(loss_values, weight=float(halo_maps.shape[0]))
            if global_step % config.training.log_interval == 0 or global_step == 1:
                record = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    **loss_values,
                }
                append_jsonl(metrics_path, record)
                print(
                    f"epoch={epoch + 1}/{config.training.epochs} step={global_step} "
                    f"loss={loss_values['loss']:.5f} bce={loss_values['bce']:.5f} "
                    f"dice={loss_values['dice']:.5f}"
                )

            if config.training.max_steps is not None and global_step >= config.training.max_steps:
                stop_training = True
                break

        train_metrics = epoch_accumulator.compute()
        train_record = {
            "event": "train_epoch",
            "epoch": epoch,
            "step": global_step,
            "seconds": time.time() - start_time,
            **train_metrics,
        }
        append_jsonl(metrics_path, train_record)

        should_evaluate = (epoch + 1) % config.training.eval_interval_epochs == 0 or stop_training
        if should_evaluate:
            visualization_path = output_dir / "visualizations" / f"epoch_{epoch + 1:03d}.png"
            val_metrics = evaluate_model(
                model,
                val_loader,
                criterion,
                device,
                include_reachability=True,
                visualization_path=visualization_path if config.training.save_visualizations > 0 else None,
                visualization_count=config.training.save_visualizations,
            )
            append_jsonl(
                metrics_path,
                {"event": "validation", "epoch": epoch, "step": global_step, **val_metrics},
            )
            print(
                f"validation loss={val_metrics['loss']:.5f} occupied_iou={val_metrics['occupied_iou']:.4f} "
                f"boundary_f1={val_metrics['boundary_f1']:.4f} reachability_iou={val_metrics['reachability_iou']:.4f}"
            )
            if val_metrics["loss"] < best_metric:
                best_metric = val_metrics["loss"]
                atomic_torch_save(
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                    ),
                    best_checkpoint,
                )

        if (epoch + 1) % config.training.checkpoint_interval_epochs == 0 or stop_training:
            atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                ),
                output_dir / f"epoch_{epoch + 1:03d}.pt",
            )
        if stop_training:
            break

    atomic_torch_save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
        ),
        last_checkpoint,
    )
    summary = {
        "best_metric": best_metric,
        "global_step": global_step,
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return TrainResult(
        output_dir=output_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        best_metric=best_metric,
        global_step=global_step,
    )
