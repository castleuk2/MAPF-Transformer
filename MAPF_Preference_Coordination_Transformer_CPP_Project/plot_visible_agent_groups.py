from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "visible_agent_distribution"
DATA = json.loads((REPORT / "visible_agent_distribution.json").read_text())

plt.rcParams.update({
    "font.family": "Noto Sans CJK KR",
    "axes.unicode_minus": False,
    "font.size": 11,
})

COLORS = {"Train": "#2563EB", "Balanced Val": "#EF4444"}
GROUPS = [
    ("maze_n16", "Maze · 16 Agents"),
    ("maze_n24", "Maze · 24 Agents"),
    ("maze_n32", "Maze · 32 Agents"),
    ("random_n16", "Random · 16 Agents"),
    ("random_n24", "Random · 24 Agents"),
    ("random_n32", "Random · 32 Agents"),
]


def series(split: str, group: str, maximum: int) -> np.ndarray:
    values = DATA[split]["groups"][group]["percent_including_ego"]
    return np.asarray([float(values.get(str(i), 0.0)) for i in range(1, maximum + 1)])


for index, (group, title) in enumerate(GROUPS, start=1):
    train = DATA["train"]["groups"][group]
    val = DATA["balanced_val_65536"]["groups"][group]
    maximum = max(train["max_including_ego"], val["max_including_ego"])
    x = np.arange(1, maximum + 1)

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.plot(x, series("train", group, maximum), marker="o", markersize=4,
            linewidth=2.2, color=COLORS["Train"], label="Train")
    ax.plot(x, series("balanced_val_65536", group, maximum), marker="o", markersize=4,
            linewidth=2.2, color=COLORS["Balanced Val"], label="Balanced Val")
    ax.set_title(f"{title}: Ego 중심 15×15 내부 Agent 수 분포", fontsize=15, weight="bold")
    ax.set_xlabel("한 Ego·한 시점의 15×15 내부 Agent 수 (Ego 포함)")
    ax.set_ylabel("해당 Agent 수를 가진 Ego-step 샘플의 비율 (%)")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.995, 0.98,
        "각 선 내부에서 전체 Ego-step 샘플 = 100%",
        transform=ax.transAxes, ha="right", va="top", color="#4B5563", fontsize=10,
    )
    fig.tight_layout()
    stem = f"{index + 2:02d}_{group}_distribution"
    for suffix in ("png", "pdf"):
        fig.savefig(REPORT / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

print(REPORT)
