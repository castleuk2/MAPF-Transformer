from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "mapf_transformer": "MAPF Transformer",
    "mapf_lns2": "MAPF-LNS2",
}
COLORS = {
    "mapf_transformer": "#4C78A8",
    "mapf_lns2": "#E45756",
}


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    print(base.with_suffix(".png"))
    print(base.with_suffix(".svg"))


def plot_success_rate(data: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, constrained_layout=True)
    for axis, suite in zip(axes, ["random", "mazes"]):
        subset = data[data["suite"] == suite]
        for model in ["mapf_transformer", "mapf_lns2"]:
            rows = subset[subset["model"] == model].sort_values("num_agents")
            values = rows["SR"].to_numpy(dtype=float) * 100
            axis.plot(
                rows["num_agents"], values, "o-", linewidth=2.5, markersize=7,
                color=COLORS[model], label=MODEL_LABELS[model],
            )
            for x, y, successes, episodes in zip(
                rows["num_agents"], values, rows["successes"], rows["episodes"]
            ):
                offset = -17 if model == "mapf_lns2" else 9
                axis.annotate(
                    f"{y:.0f}% ({int(successes)}/{int(episodes)})",
                    (x, y), xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=9, color=COLORS[model],
                )
        axis.set_title("Random maps" if suite == "random" else "Maze maps", fontsize=13)
        axis.set_xlabel("Number of agents")
        axis.set_xticks([8, 16, 24])
        axis.set_ylim(-4, 108)
    axes[0].set_ylabel("Success Rate, SR (%)")
    axes[1].legend(loc="lower left")
    fig.suptitle("Success rate drops as the number of agents increases", fontsize=16)
    save_figure(fig, output_dir / "sr_by_map_agent")
    plt.close(fig)


def plot_both_success(data: pd.DataFrame, output_dir: Path) -> None:
    rows = data[(data["suite"].isin(["random", "mazes"])) & (data["num_agents"] != "all")].copy()
    rows["num_agents"] = rows["num_agents"].astype(int)
    order = [(suite, agents) for suite in ["random", "mazes"] for agents in [8, 16, 24]]
    rows = rows.set_index(["suite", "num_agents"]).loc[order].reset_index()
    labels = [f"{suite.title()}\n{agents} agents" for suite, agents in order]
    counts = rows["both_success_episodes"].astype(int).to_numpy()
    x = np.arange(len(rows), dtype=float)
    width = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    specs = [
        ("SoC", "Sum of Costs (SoC)"),
        ("makespan", "Makespan"),
        ("runtime", "Runtime (seconds)"),
    ]
    for axis, (metric, title) in zip(axes, specs):
        transformer = rows[f"transformer_{metric}"].to_numpy(dtype=float)
        lns2 = rows[f"mapf_lns2_{metric}"].to_numpy(dtype=float)
        left = axis.bar(
            x - width / 2, transformer, width, color=COLORS["mapf_transformer"],
            label=MODEL_LABELS["mapf_transformer"],
        )
        right = axis.bar(
            x + width / 2, lns2, width, color=COLORS["mapf_lns2"],
            label=MODEL_LABELS["mapf_lns2"],
        )
        axis.bar_label(left, fmt="%.2f", fontsize=8, padding=2, rotation=90)
        axis.bar_label(right, fmt="%.2f", fontsize=8, padding=2, rotation=90)
        axis.set_title(title, fontsize=13)
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_ylim(0, max(np.max(transformer), np.max(lns2)) * 1.27)
        for index, count in enumerate(counts):
            axis.text(
                index, axis.get_ylim()[1] * 0.95, f"n={count}",
                ha="center", va="top", fontsize=9, fontweight="bold",
            )
    axes[0].set_ylabel("Mean over episodes where both methods achieved SR=1")
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Paired comparison on episodes successfully solved by both methods", fontsize=16)
    save_figure(fig, output_dir / "both_success_metrics_by_map_agent")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPF Transformer vs MAPF-LNS2 benchmark")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    plot_success_rate(pd.read_csv(args.result_dir / "sr_by_map_agent.csv"), args.result_dir)
    plot_both_success(pd.read_csv(args.result_dir / "both_success_comparison.csv", dtype={"num_agents": str}), args.result_dir)


if __name__ == "__main__":
    main()
