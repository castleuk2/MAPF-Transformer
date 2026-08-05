from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mapf_map_transformer.config import experiment_config_from_dict
from mapf_map_transformer.model import StructuredMapTransformer
from mapf_map_transformer.policy_data import EpisodeBatchSampler, PolicyHistoryHaloDataset


@dataclass
class Counter:
    maps: int = 0
    cells: int = 0
    ce_sum: float = 0.0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    exact_maps: int = 0
    error_cells: int = 0
    max_error_cells: int = 0

    def update(self, losses, prediction, target) -> None:
        errors = prediction.ne(target).flatten(1).sum(1)
        self.maps += int(target.shape[0])
        self.cells += int(target.numel())
        self.ce_sum += float(losses.sum().item())
        self.tp += int(((prediction == 1) & (target == 1)).sum().item())
        self.tn += int(((prediction == 0) & (target == 0)).sum().item())
        self.fp += int(((prediction == 1) & (target == 0)).sum().item())
        self.fn += int(((prediction == 0) & (target == 1)).sum().item())
        self.exact_maps += int(errors.eq(0).sum().item())
        self.error_cells += int(errors.sum().item())
        self.max_error_cells = max(self.max_error_cells, int(errors.max().item()))

    def metrics(self) -> dict:
        failed = self.maps - self.exact_maps
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        return {
            **asdict(self), "failed_maps": failed,
            "cell_ce": self.ce_sum / max(self.cells, 1),
            "cell_accuracy": (self.tp + self.tn) / max(self.cells, 1),
            "obstacle_precision": precision,
            "obstacle_recall": recall,
            "obstacle_f1": 2 * precision * recall / max(precision + recall, 1e-30),
            "obstacle_iou": self.tp / max(self.tp + self.fp + self.fn, 1),
            "exact_map_rate": self.exact_maps / max(self.maps, 1),
            "mean_error_cells_per_failed_map": self.error_cells / max(failed, 1),
        }


@torch.inference_mode()
def evaluate(model, loader, device) -> dict:
    counters = {"overall": Counter(), "maze": Counter(), "random": Counter()}
    model.eval()
    for batch_index, batch in enumerate(loader, 1):
        valid = batch["frame_valid"].bool()
        maps = batch["halo_maps"][valid].to(device=device, dtype=torch.long, non_blocking=True)
        families = batch["map_family"].unsqueeze(1).expand_as(valid)[valid]
        target = maps[:, 1:-1, 1:-1]
        logits = model(maps).reconstruction_logits
        losses = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1), reduction="none").reshape_as(target)
        prediction = logits.argmax(-1)
        counters["overall"].update(losses, prediction, target)
        for family_id, name in ((0, "maze"), (1, "random")):
            mask = families.eq(family_id).to(device)
            if mask.any():
                counters[name].update(losses[mask], prediction[mask], target[mask])
        if batch_index % 200 == 0:
            print(f"batches={batch_index}/{len(loader)} maps={counters['overall'].maps:,}", flush=True)
    return {name: counter.metrics() for name, counter in counters.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "eval"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = experiment_config_from_dict(checkpoint["config"])
    device = torch.device(args.device)
    model = StructuredMapTransformer(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    results = {}
    paths = {"train": config.dataset.train_path, "val": config.dataset.val_path, "eval": config.dataset.eval_path}
    for split in args.splits:
        dataset = PolicyHistoryHaloDataset(
            paths[split], history_frames=config.dataset.history_frames,
            min_history_frames=config.dataset.min_history_frames,
            goal_wait_keep_ratio=config.dataset.goal_wait_keep_ratio,
            history_augmentation=False, seed=config.dataset.seed + 1,
        )
        sampler = EpisodeBatchSampler(dataset, args.batch_size, False, config.training.seed + 1)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
        print(f"split={split} policy_samples={len(dataset):,}")
        results[split] = evaluate(model, loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        keys = ["split", "map_type", *next(iter(results.values()))["overall"].keys()]
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for split, groups in results.items():
            for map_type, values in groups.items():
                writer.writerow({"split": split, "map_type": map_type, **values})
    print(f"json={args.output}\ncsv={csv_path}")


if __name__ == "__main__":
    main()
