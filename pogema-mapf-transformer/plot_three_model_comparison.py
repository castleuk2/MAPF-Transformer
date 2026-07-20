from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["mapf_transformer", "mapf_gpt_6m", "mapf_lns2"]
LABELS = {
    "mapf_transformer": "MAPF-Transformer",
    "mapf_gpt_6m": "MAPF-GPT-6M",
    "mapf_lns2": "MAPF-LNS2",
}
COLORS = {
    "mapf_transformer": "#4C78A8",
    "mapf_gpt_6m": "#F58518",
    "mapf_lns2": "#54A24B",
}
SUITES = ["random", "mazes"]
AGENTS = [8, 16, 24]
KEYS = ["suite", "map_name", "seed", "num_agents"]


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def validate_and_merge(learned_path: Path, lns2_path: Path) -> pd.DataFrame:
    learned = pd.read_csv(learned_path)
    lns2 = pd.read_csv(lns2_path)
    lns2 = lns2[lns2.model == "mapf_lns2"].copy()
    expected = set(map(tuple, learned[KEYS].drop_duplicates().to_numpy()))
    actual = set(map(tuple, lns2[KEYS].to_numpy()))
    if expected != actual:
        raise ValueError(
            f"Scenario mismatch: missing LNS2={len(expected - actual)}, "
            f"extra LNS2={len(actual - expected)}"
        )
    return pd.concat([learned, lns2], ignore_index=True)


def aggregate(data: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sr = (
        data.groupby(["model", "suite", "num_agents"], as_index=False)
        .agg(episodes=("CSR", "size"), successes=("CSR", "sum"), SR=("CSR", "mean"))
    )
    sr.to_csv(output / "three_models_sr_by_map_agent.csv", index=False)

    indexed = {model: data[data.model == model].set_index(KEYS) for model in MODELS}
    rows, episode_rows = [], []
    for suite in SUITES:
        for agents in AGENTS:
            base_keys = [
                key for key in indexed[MODELS[0]].index
                if key[0] == suite and int(key[3]) == agents
            ]
            common = [
                key for key in base_keys
                if all(float(indexed[model].loc[key, "CSR"]) == 1.0 for model in MODELS)
            ]
            row = {"suite": suite, "num_agents": agents, "all_success_episodes": len(common)}
            for model in MODELS:
                for metric in ("SoC", "makespan", "runtime"):
                    row[f"{model}_{metric}"] = (
                        float(np.mean([float(indexed[model].loc[key, metric]) for key in common]))
                        if common else np.nan
                    )
            rows.append(row)
            for key in common:
                item = dict(zip(KEYS, key))
                for model in MODELS:
                    for metric in ("SoC", "makespan", "runtime"):
                        item[f"{model}_{metric}"] = float(indexed[model].loc[key, metric])
                episode_rows.append(item)
    paired = pd.DataFrame(rows)
    paired.to_csv(output / "three_models_all_success_comparison.csv", index=False)
    pd.DataFrame(episode_rows).to_csv(output / "three_models_all_success_episodes.csv", index=False)
    return sr, paired


def plot_sr(data: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True, constrained_layout=True)
    x = np.arange(len(AGENTS))
    width = 0.25
    offsets = [-width, 0, width]
    for axis, suite in zip(axes, SUITES):
        for offset, model in zip(offsets, MODELS):
            rows = data[(data.suite == suite) & (data.model == model)].set_index("num_agents").loc[AGENTS]
            values = rows.SR.to_numpy(float) * 100
            bars = axis.bar(x + offset, values, width, color=COLORS[model], label=LABELS[model])
            axis.bar_label(
                bars,
                labels=[f"{v:.0f}%\n({int(s)}/50)" for v, s in zip(values, rows.successes)],
                padding=3, fontsize=8,
            )
        axis.set_title("Random maps" if suite == "random" else "Maze maps")
        axis.set_xlabel("Number of agents")
        axis.set_xticks(x, AGENTS)
        axis.set_ylim(0, 116)
    axes[0].set_ylabel("Success Rate, SR (%)")
    axes[1].legend(loc="lower left")
    fig.suptitle("Success Rate: MAPF-Transformer vs MAPF-GPT-6M vs MAPF-LNS2", fontsize=15)
    save(fig, output / "three_models_success_rate_comparison")
    plt.close(fig)


def plot_paired(data: pd.DataFrame, output: Path) -> None:
    order = [(suite, agents) for suite in SUITES for agents in AGENTS]
    data = data.set_index(["suite", "num_agents"]).loc[order].reset_index()
    labels = [f"{'Random' if s == 'random' else 'Maze'}\n{a} agents" for s, a in order]
    x = np.arange(len(data))
    width = 0.25
    offsets = [-width, 0, width]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("SoC", "makespan", "runtime"), ("Sum of Costs (SoC)", "Makespan", "Runtime (seconds)")
    ):
        maximum = 0.0
        for offset, model in zip(offsets, MODELS):
            values = data[f"{model}_{metric}"].to_numpy(float)
            maximum = max(maximum, float(np.nanmax(values)))
            bars = axis.bar(x + offset, values, width, color=COLORS[model], label=LABELS[model])
            axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, rotation=90)
        axis.set_title(title)
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_ylim(0, maximum * 1.30)
        for idx, count in enumerate(data.all_success_episodes):
            axis.text(idx, maximum * 1.24, f"n={int(count)}", ha="center", va="top", fontsize=9)
    axes[0].set_ylabel("Mean on episodes solved by all three methods")
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Paired comparison where all three methods achieved SR=1", fontsize=15)
    save(fig, output / "three_models_all_success_metrics")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("learned_results", type=Path)
    parser.add_argument("lns2_results", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = validate_and_merge(
        args.learned_results / "episodes.csv",
        args.lns2_results / "episodes.csv",
    )
    merged.to_csv(args.output_dir / "three_models_episodes.csv", index=False)
    sr, paired = aggregate(merged, args.output_dir)
    plot_sr(sr, args.output_dir)
    plot_paired(paired, args.output_dir)


if __name__ == "__main__":
    main()
