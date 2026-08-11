from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def trunc4(value: float) -> str:
    return f"{math.floor(value * 10_000) / 10_000:.4f}"


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
args = parser.parse_args()
rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
if not rows or any(f"train_total" not in row for row in rows):
    raise RuntimeError("Epoch별 train_total과 val_total이 모두 기록된 metrics.jsonl이 필요합니다.")

plt.rcParams.update({
    "font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 11,
})
epochs = [row["epoch"] for row in rows]
colors = {"Train": "#2563EB", "Validation": "#EF4444"}


def line(ax, train_key: str, val_key: str, title: str) -> None:
    for label, key in (("Train", train_key), ("Validation", val_key)):
        values = [row[key] for row in rows]
        ax.plot(epochs, values, marker="o", linewidth=2.4, markersize=7,
                label=label, color=colors[label])
        for x, value in zip(epochs, values):
            ax.annotate(trunc4(value), (x, value), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_title(title, weight="bold", pad=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_xticks(epochs); ax.grid(alpha=0.25); ax.legend(frameon=False)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
line(axes[0], "train_total", "val_total", "MPCT Total Loss")
line(axes[1], "train_action", "val_action", "MPCT Action Loss")
fig.suptitle("MPCT Train / Validation Loss", fontsize=16, weight="bold")
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(args.run_dir / f"train_val_loss.{suffix}", dpi=180, bbox_inches="tight")
plt.close(fig)

components = [
    ("ego_action", "Ego Action CE"), ("neighbor_action", "Neighbor Action CE"),
    ("act_request", "ACT Request CE"), ("conflict", "Conflict CE"),
    ("reason", "Reason BCE"), ("scene_risk", "Scene-risk CE"),
]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (key, title) in zip(axes.flat, components):
    line(ax, f"train_{key}", f"val_{key}", title)
fig.suptitle("MPCT Loss Components", fontsize=16, weight="bold")
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(args.run_dir / f"train_val_loss_components.{suffix}", dpi=180, bbox_inches="tight")
plt.close(fig)
print(args.run_dir)
