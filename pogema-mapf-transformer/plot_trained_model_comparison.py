from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "mapf_transformer": "MAPF-Transformer",
    "mapf_gpt_6m": "MAPF-GPT-6M",
}
COLORS = {
    "mapf_transformer": "#4C78A8",
    "mapf_gpt_6m": "#F58518",
}
MODELS = ["mapf_transformer", "mapf_gpt_6m"]
SUITES = ["random", "mazes"]
AGENTS = [8, 16, 24]


def save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def aggregate(episodes: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sr = (
        episodes.groupby(["model", "suite", "num_agents"], as_index=False)
        .agg(episodes=("CSR", "size"), successes=("CSR", "sum"), SR=("CSR", "mean"))
    )
    sr.to_csv(output / "sr_by_map_agent.csv", index=False)

    keyed = ["suite", "map_name", "seed", "num_agents"]
    left = episodes[episodes.model == MODELS[0]].set_index(keyed)
    right = episodes[episodes.model == MODELS[1]].set_index(keyed)
    common = left.index.intersection(right.index)
    rows = []
    paired_episode_rows = []
    for suite in SUITES:
        for agents in AGENTS:
            keys = [key for key in common if key[0] == suite and key[3] == agents]
            both = [key for key in keys if float(left.loc[key, "CSR"]) == 1 and float(right.loc[key, "CSR"]) == 1]
            for key in both:
                paired_episode_rows.append(
                    {
                        "suite": key[0],
                        "map_name": key[1],
                        "seed": key[2],
                        "num_agents": key[3],
                        **{
                            f"{model}_{metric}": float(
                                (left if model == MODELS[0] else right).loc[key, metric]
                            )
                            for model in MODELS
                            for metric in ("SoC", "makespan", "runtime")
                        },
                    }
                )
            row = {"suite": suite, "num_agents": agents, "both_success_episodes": len(both)}
            for model, frame in ((MODELS[0], left), (MODELS[1], right)):
                for metric in ("SoC", "makespan", "runtime"):
                    row[f"{model}_{metric}"] = (
                        float(np.mean([float(frame.loc[key, metric]) for key in both]))
                        if both
                        else np.nan
                    )
            rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(output / "both_success_comparison.csv", index=False)
    pd.DataFrame(paired_episode_rows).to_csv(output / "both_success_episodes.csv", index=False)
    return sr, paired


def plot_sr(data: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, constrained_layout=True)
    for axis, suite in zip(axes, SUITES):
        for model in MODELS:
            rows = data[(data.suite == suite) & (data.model == model)].sort_values("num_agents")
            y = rows.SR.to_numpy() * 100
            axis.plot(rows.num_agents, y, "o-", lw=2.7, ms=7, color=COLORS[model], label=LABELS[model])
            for x, value, successes, total in zip(rows.num_agents, y, rows.successes, rows.episodes):
                offset = 10 if model == MODELS[1] else -18
                axis.annotate(
                    f"{value:.0f}% ({int(successes)}/{int(total)})",
                    (x, value), xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=9, color=COLORS[model],
                )
        axis.set_title("Random maps" if suite == "random" else "Maze maps")
        axis.set_xlabel("Number of agents")
        axis.set_xticks(AGENTS)
        axis.set_ylim(-3, 105)
    axes[0].set_ylabel("Success Rate, SR (%)")
    axes[1].legend()
    fig.suptitle("Success rate by map type and number of agents", fontsize=15)
    save(fig, output / "sr_by_map_agent")
    plt.close(fig)


def plot_sr_bars(data: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True, constrained_layout=True)
    x = np.arange(len(AGENTS))
    width = 0.36
    for axis, suite in zip(axes, SUITES):
        for index, model in enumerate(MODELS):
            rows = data[(data.suite == suite) & (data.model == model)].set_index("num_agents").loc[AGENTS]
            values = rows.SR.to_numpy(float) * 100
            bars = axis.bar(
                x + (-width / 2 if index == 0 else width / 2),
                values,
                width,
                color=COLORS[model],
                label=LABELS[model],
            )
            labels = [
                f"{value:.0f}%\n({int(success)}/50)"
                for value, success in zip(values, rows.successes)
            ]
            axis.bar_label(bars, labels=labels, padding=3, fontsize=9)
        axis.set_title("Random maps" if suite == "random" else "Maze maps")
        axis.set_xlabel("Number of agents")
        axis.set_xticks(x, AGENTS)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Success Rate, SR (%)")
    axes[1].legend()
    fig.suptitle("MAPF-Transformer vs MAPF-GPT-6M: Success Rate", fontsize=15)
    save(fig, output / "success_rate_comparison")
    plt.close(fig)


def plot_paired(data: pd.DataFrame, output: Path) -> None:
    order = [(suite, agents) for suite in SUITES for agents in AGENTS]
    data = data.set_index(["suite", "num_agents"]).loc[order].reset_index()
    labels = [f"{'Random' if s == 'random' else 'Maze'}\n{a} agents" for s, a in order]
    x = np.arange(len(data))
    width = 0.36
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("SoC", "makespan", "runtime"), ("Sum of Costs (SoC)", "Makespan", "Runtime (seconds)")
    ):
        maxima = 0.0
        for index, model in enumerate(MODELS):
            values = data[f"{model}_{metric}"].to_numpy(float)
            maxima = max(maxima, float(np.nanmax(values)))
            bars = axis.bar(
                x + (-width / 2 if index == 0 else width / 2),
                values, width, color=COLORS[model], label=LABELS[model],
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, rotation=90)
        axis.set_title(title)
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_ylim(0, maxima * 1.30)
        for idx, count in enumerate(data.both_success_episodes):
            axis.text(idx, maxima * 1.24, f"n={int(count)}", ha="center", va="top", fontsize=9)
    axes[0].set_ylabel("Mean on episodes solved by both models")
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Paired comparison where both models achieved SR=1", fontsize=15)
    save(fig, output / "both_success_metrics_by_map_agent")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    episodes = pd.read_csv(args.result_dir / "episodes.csv")
    sr, paired = aggregate(episodes, args.result_dir)
    plot_sr(sr, args.result_dir)
    plot_sr_bars(sr, args.result_dir)
    plot_paired(paired, args.result_dir)


if __name__ == "__main__":
    main()
