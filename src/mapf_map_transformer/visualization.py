from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_reconstruction_grid(
    halo_maps: torch.Tensor,
    reconstruction_logits: torch.Tensor,
    output_path: str | Path,
    *,
    max_items: int = 8,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    halo = halo_maps.detach().cpu().numpy()
    probability = torch.sigmoid(reconstruction_logits).detach().cpu().numpy()
    count = min(max_items, len(halo))
    if count <= 0:
        return

    figure, axes = plt.subplots(count, 3, figsize=(8, 2.7 * count), squeeze=False)
    for index in range(count):
        core = halo[index, 1:-1, 1:-1]
        predicted = probability[index]
        error = np.abs(predicted - core.astype(np.float32))
        axes[index, 0].imshow(core, vmin=0, vmax=1, interpolation="nearest")
        axes[index, 0].set_title("Target occupied map")
        axes[index, 1].imshow(predicted, vmin=0, vmax=1, interpolation="nearest")
        axes[index, 1].set_title("Reconstruction probability")
        axes[index, 2].imshow(error, vmin=0, vmax=1, interpolation="nearest")
        axes[index, 2].set_title("Absolute error")
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
