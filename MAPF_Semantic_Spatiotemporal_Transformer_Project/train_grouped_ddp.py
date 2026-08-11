from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from mapf_sst.checkpoint import save_checkpoint
from mapf_sst.communication import MultiRoundCommunicationPolicy
from mapf_sst.config import load_config
from mapf_sst.data.npz_dataset import EpisodeFeatureBuilder
from mapf_sst.losses import compute_loss
from mapf_sst.model import SemanticSpatiotemporalPolicy
from train import seed_everything
from mapf_sst.types import concatenate_communication_graphs, concatenate_policy_batches


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Frame:
    path: Path
    time_step: int
    family: str
    agents: int


def read_frames(manifest: str | Path) -> list[Frame]:
    manifest = Path(manifest)
    frames: list[Frame] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        path = Path(record["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        for time_step in range(int(record["time_steps"])):
            frames.append(Frame(path.resolve(), time_step, str(record["map_family"]), int(record["num_agents"])))
    return frames


def uniform_indices(total: int, count: int, *, offset: int = 0) -> list[int]:
    count = min(total, count)
    if count == total:
        return list(range(total))
    values = np.linspace(0, total - 1, num=count, dtype=np.int64)
    return [int((value + offset) % total) for value in values]


def balanced_validation(frames: list[Frame], count: int) -> list[Frame]:
    groups = {(family, agents): [] for family in ("maze", "random") for agents in (16, 24, 32)}
    for frame in frames:
        groups[(frame.family, frame.agents)].append(frame)
    base, remainder = divmod(count, len(groups))
    selected: list[Frame] = []
    for position, key in enumerate(groups):
        pool = groups[key]
        quota = base + int(position < remainder)
        selected.extend(pool[index] for index in uniform_indices(len(pool), quota))
    return selected


def grouped_metrics(model, frames, builder, config, device, rounds):
    model.eval()
    total_loss = correct = examples = 0.0
    with torch.no_grad():
        frame_batch = max(1, config.training.val_batch_size)
        for start in range(0, len(frames), frame_batch):
            pairs = []
            for frame in frames[start : start + frame_batch]:
                episode = builder.load_episode(frame.path)
                pairs.append(builder.build_all_views(episode, frame.time_step))
            counts = [pair[0].batch_size for pair in pairs]
            batch = concatenate_policy_batches([pair[0] for pair in pairs])
            graph = concatenate_communication_graphs([pair[1] for pair in pairs], counts)
            batch, graph = batch.to(device), graph.to(device)
            output = model(batch, graph, rounds=rounds).final
            loss = compute_loss(output, batch, config.model, config.training)
            n = batch.batch_size
            total_loss += float(loss.total) * n
            correct += float((output.ego_logits.argmax(-1) == batch.ego_action).sum())
            examples += n
    return {"val_total": total_loss / examples, "val_ego_accuracy": correct / examples, "val_ego_samples": int(examples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-train-frames", type=int, default=None, help="smoke-test override")
    parser.add_argument("--max-val-frames", type=int, default=None, help="smoke-test override")
    parser.add_argument("--epochs", type=int, default=None, help="smoke-test override")
    parser.add_argument("--output-dir", type=Path, default=None, help="output override")
    args = parser.parse_args()
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    config = load_config(args.config)
    rounds = config.training.communication_rounds
    if rounds != 4:
        raise ValueError("this reproducible ablation config requires communication_rounds=4")
    seed_everything(config.training.seed + rank)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (ROOT / config.training.output_dir).resolve()
    )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")

    train_frames = read_frames(config.data.train_manifest)
    train_frame_count = args.max_train_frames or config.data.grouped_train_frames_per_epoch
    val_frame_count = args.max_val_frames or config.data.grouped_val_frames
    epochs = args.epochs or config.training.epochs
    val_frames = balanced_validation(read_frames(config.data.val_manifest), val_frame_count)
    if config.data.feature_backend == "cpp":
        from mapf_sst.cpp import CppEpisodeFeatureBuilder
        builder = CppEpisodeFeatureBuilder(
            config.model, coordinate_order=config.data.coordinate_order
        )
    elif config.data.feature_backend == "python":
        builder = EpisodeFeatureBuilder(
            config.model, coordinate_order=config.data.coordinate_order
        )
    else:
        raise ValueError(f"unknown feature_backend: {config.data.feature_backend}")
    base = SemanticSpatiotemporalPolicy(config.model).to(device)
    if not base.map_encoder.is_frozen:
        raise RuntimeError("pretrained Map must be frozen")
    wrapper = MultiRoundCommunicationPolicy(base).to(device)
    model = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.training.learning_rate, weight_decay=config.training.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp)
    best, step = float("inf"), 0

    for epoch in range(epochs):
        model.train()
        chosen = uniform_indices(len(train_frames), train_frame_count, offset=epoch * 104729)
        usable = len(chosen) - len(chosen) % world
        local_indices = chosen[rank:usable:world]
        sums = torch.zeros(4, device=device, dtype=torch.float64)
        frame_batch = max(1, config.training.batch_size)
        log_time = time.perf_counter()
        for start in range(0, len(local_indices), frame_batch):
            indices = local_indices[start : start + frame_batch]
            pairs = []
            for index in indices:
                frame = train_frames[index]
                episode = builder.load_episode(frame.path)
                pairs.append(builder.build_all_views(episode, frame.time_step))
            counts = [pair[0].batch_size for pair in pairs]
            batch = concatenate_policy_batches([pair[0] for pair in pairs])
            graph = concatenate_communication_graphs([pair[1] for pair in pairs], counts)
            batch, graph = batch.to(device), graph.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config.training.amp):
                output = model(batch, graph, rounds=rounds).final
                losses = compute_loss(output, batch, config.model, config.training)
            local_n = torch.tensor(float(batch.batch_size), device=device)
            global_n = local_n.clone(); dist.all_reduce(global_n)
            scaled = losses.total * (world * local_n / global_n)
            scaler.scale(scaled).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.training.grad_clip_norm)
            scaler.step(optimizer); scaler.update(); step += 1
            sums[0] += losses.total.detach().double() * local_n
            sums[1] += (output.ego_logits.argmax(-1) == batch.ego_action).sum()
            sums[2] += local_n
            sums[3] += len(indices)
            if rank == 0 and step % 100 == 0:
                now = time.perf_counter()
                print(
                    f"epoch={epoch+1} step={step} local_ego={batch.batch_size} "
                    f"loss={float(losses.total):.6f} grouped_steps_per_s={100/(now-log_time):.2f}",
                    flush=True,
                )
                log_time = now
        dist.all_reduce(sums)
        if rank == 0:
            val = grouped_metrics(wrapper, val_frames, builder, config, device, rounds)
            event = {"epoch": epoch+1, "step": step, "rounds": rounds, "train_total": float(sums[0]/sums[2]), "train_ego_accuracy": float(sums[1]/sums[2]), "train_ego_samples": int(sums[2]), **val}
            print(json.dumps(event, ensure_ascii=False), flush=True)
            with (output_dir/"metrics.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(event, ensure_ascii=False)+"\n")
            path = output_dir/f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(path, model=base, optimizer=optimizer, config=config, epoch=epoch+1, step=step, metrics=event)
            shutil.copy2(path, output_dir/"last.pt")
            if val["val_total"] < best: best=val["val_total"]; shutil.copy2(path, output_dir/"best.pt")
            if epoch+1 in {3,6}:
                shutil.copy2(path, output_dir/f"last_epoch{epoch+1}.pt")
                shutil.copy2(output_dir/"best.pt", output_dir/f"best_epoch{epoch+1}.pt")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
