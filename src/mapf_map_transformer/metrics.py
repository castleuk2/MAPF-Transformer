from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import extract_core, patchify_core
from .losses import occupancy_boundary_mask


@torch.no_grad()
def binary_reconstruction_metrics(
    logits: torch.Tensor,
    halo_maps: torch.Tensor,
    *,
    occupied_state: int = 1,
    threshold: float = 0.5,
    boundary_tolerance: int = 1,
    include_reachability: bool = True,
) -> dict[str, float]:
    if halo_maps.ndim == 2:
        halo_maps = halo_maps.unsqueeze(0)
    target = extract_core(halo_maps).eq(occupied_state)
    prediction = torch.sigmoid(logits).ge(threshold)
    target = target.to(device=prediction.device)

    tp = (prediction & target).sum().item()
    tn = ((~prediction) & (~target)).sum().item()
    fp = (prediction & (~target)).sum().item()
    fn = ((~prediction) & target).sum().item()
    eps = 1.0e-9

    metrics = {
        "cell_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "occupied_precision": tp / max(tp + fp, eps),
        "occupied_recall": tp / max(tp + fn, eps),
        "occupied_f1": (2 * tp) / max(2 * tp + fp + fn, eps),
        "occupied_iou": tp / max(tp + fp + fn, eps),
    }

    target_patch = patchify_core(target.to(torch.uint8), patch_size=3).to(torch.bool)
    pred_patch = patchify_core(prediction.to(torch.uint8), patch_size=3).to(torch.bool)
    metrics["patch_exact_rate"] = pred_patch.eq(target_patch).all(dim=(-1, -2)).to(torch.float32).mean().item()
    metrics["map_exact_rate"] = prediction.eq(target).all(dim=(-1, -2)).to(torch.float32).mean().item()

    target_boundary = occupancy_boundary_mask(target.to(torch.float32)).to(torch.bool)
    pred_boundary = occupancy_boundary_mask(prediction.to(torch.float32)).to(torch.bool)
    if boundary_tolerance > 0:
        kernel = 2 * boundary_tolerance + 1
        target_dilated = F.max_pool2d(
            target_boundary.to(torch.float32).unsqueeze(1), kernel, stride=1, padding=boundary_tolerance
        ).squeeze(1).to(torch.bool)
        pred_dilated = F.max_pool2d(
            pred_boundary.to(torch.float32).unsqueeze(1), kernel, stride=1, padding=boundary_tolerance
        ).squeeze(1).to(torch.bool)
    else:
        target_dilated = target_boundary
        pred_dilated = pred_boundary
    matched_pred = (pred_boundary & target_dilated).sum().item()
    matched_target = (target_boundary & pred_dilated).sum().item()
    pred_count = pred_boundary.sum().item()
    target_count = target_boundary.sum().item()
    boundary_precision = matched_pred / max(pred_count, eps)
    boundary_recall = matched_target / max(target_count, eps)
    metrics["boundary_f1"] = 2 * boundary_precision * boundary_recall / max(
        boundary_precision + boundary_recall, eps
    )

    if include_reachability:
        reachability = _batch_reachability_metrics(
            prediction.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
        )
        metrics.update(reachability)
    return metrics


def _reachable_free_mask(occupied: np.ndarray, start: tuple[int, int] | None = None) -> np.ndarray:
    height, width = occupied.shape
    start = start or (height // 2, width // 2)
    reachable = np.zeros_like(occupied, dtype=bool)
    if occupied[start]:
        return reachable
    queue: deque[tuple[int, int]] = deque([start])
    reachable[start] = True
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and not occupied[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                queue.append((nr, nc))
    return reachable


def _batch_reachability_metrics(pred_occupied: np.ndarray, target_occupied: np.ndarray) -> dict[str, float]:
    ious: list[float] = []
    accuracies: list[float] = []
    border_agreements: list[float] = []
    for pred, target in zip(pred_occupied, target_occupied, strict=True):
        pred_reach = _reachable_free_mask(pred)
        target_reach = _reachable_free_mask(target)
        intersection = np.logical_and(pred_reach, target_reach).sum()
        union = np.logical_or(pred_reach, target_reach).sum()
        ious.append(float(intersection / union) if union else 1.0)
        accuracies.append(float((pred_reach == target_reach).mean()))
        pred_border = np.array(
            [pred_reach[0].any(), pred_reach[-1].any(), pred_reach[:, 0].any(), pred_reach[:, -1].any()]
        )
        target_border = np.array(
            [target_reach[0].any(), target_reach[-1].any(), target_reach[:, 0].any(), target_reach[:, -1].any()]
        )
        border_agreements.append(float((pred_border == target_border).mean()))
    return {
        "reachability_iou": float(np.mean(ious)) if ious else 0.0,
        "reachability_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "border_reachability_agreement": float(np.mean(border_agreements)) if border_agreements else 0.0,
    }


@dataclass(slots=True)
class MetricAccumulator:
    weighted_sums: dict[str, float] = field(default_factory=dict)
    total_weight: float = 0.0

    def update(self, values: dict[str, float], weight: float = 1.0) -> None:
        for key, value in values.items():
            self.weighted_sums[key] = self.weighted_sums.get(key, 0.0) + float(value) * weight
        self.total_weight += weight

    def compute(self) -> dict[str, float]:
        if self.total_weight <= 0:
            return {key: 0.0 for key in self.weighted_sums}
        return {key: value / self.total_weight for key, value in self.weighted_sums.items()}
