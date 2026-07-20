from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["suite", "map_name", "seed", "num_agents"]
SUITES = ["random", "mazes"]
AGENTS = [8, 16, 24]
VARIANTS = ["baseline", "latent32", "one_hop_ctg"]
LABELS = {
    "baseline": "Baseline (latent 16)",
    "latent32": "Map latent 32",
    "one_hop_ctg": "One-hop CTG",
}
COLORS = {"baseline": "#4C78A8", "latent32": "#E45756", "one_hop_ctg": "#54A24B"}


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def load(path: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "model" in frame:
        frame = frame[frame.model == "mapf_transformer"].copy()
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate scenario keys in {path}")
    frame["variant"] = variant
    return frame.set_index(KEYS).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three MAPF Transformer ablations")
    parser.add_argument("baseline_episodes", type=Path)
    parser.add_argument("latent32_episodes", type=Path)
    parser.add_argument("one_hop_ctg_episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "baseline": load(args.baseline_episodes, "baseline"),
        "latent32": load(args.latent32_episodes, "latent32"),
        "one_hop_ctg": load(args.one_hop_ctg_episodes, "one_hop_ctg"),
    }
    reference = frames["baseline"].index
    for variant in VARIANTS[1:]:
        if not reference.equals(frames[variant].index):
            raise ValueError(f"Scenario mismatch between baseline and {variant}")

    episodes = pd.concat([frames[v].reset_index() for v in VARIANTS], ignore_index=True)
    episodes.to_csv(args.output_dir / "episodes.csv", index=False)
    sr = (
        episodes.groupby(["variant", "suite", "num_agents"], as_index=False)
        .agg(episodes=("CSR", "size"), successes=("CSR", "sum"), SR=("CSR", "mean"), ISR=("ISR", "mean"))
    )
    sr.to_csv(args.output_dir / "sr_by_map_agent.csv", index=False)

    overlap_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    paired_episode_rows: list[dict[str, object]] = []
    for suite in SUITES:
        for agents in AGENTS:
            keys = [key for key in reference if key[0] == suite and int(key[3]) == agents]
            patterns: dict[str, int] = {}
            all_success = []
            for key in keys:
                flags = [int(float(frames[v].loc[key, "CSR"]) == 1.0) for v in VARIANTS]
                pattern = "".join(map(str, flags))
                patterns[pattern] = patterns.get(pattern, 0) + 1
                if all(flags):
                    all_success.append(key)
            for pattern in sorted(patterns):
                overlap_rows.append(
                    {
                        "suite": suite,
                        "num_agents": agents,
                        "baseline_success": int(pattern[0]),
                        "latent32_success": int(pattern[1]),
                        "one_hop_ctg_success": int(pattern[2]),
                        "episodes": patterns[pattern],
                    }
                )
            row: dict[str, object] = {
                "suite": suite,
                "num_agents": agents,
                "all_success_episodes": len(all_success),
            }
            for variant in VARIANTS:
                for metric in ("SoC", "makespan", "runtime"):
                    row[f"{variant}_{metric}"] = (
                        float(np.mean([float(frames[variant].loc[key, metric]) for key in all_success]))
                        if all_success else np.nan
                    )
            paired_rows.append(row)
            for key in all_success:
                item = dict(zip(KEYS, key))
                for variant in VARIANTS:
                    for metric in ("SoC", "makespan", "runtime"):
                        item[f"{variant}_{metric}"] = float(frames[variant].loc[key, metric])
                paired_episode_rows.append(item)

    overlap = pd.DataFrame(overlap_rows)
    paired = pd.DataFrame(paired_rows)
    overlap.to_csv(args.output_dir / "success_overlap.csv", index=False)
    paired.to_csv(args.output_dir / "all_success_comparison.csv", index=False)
    pd.DataFrame(paired_episode_rows).to_csv(args.output_dir / "all_success_episodes.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, constrained_layout=True)
    offsets = {"baseline": -18, "latent32": 9, "one_hop_ctg": 23}
    for axis, suite in zip(axes, SUITES):
        for variant in VARIANTS:
            rows = sr[(sr.variant == variant) & (sr.suite == suite)].set_index("num_agents").loc[AGENTS]
            values = rows.SR.to_numpy(float) * 100
            axis.plot(AGENTS, values, "o-", lw=2.4, color=COLORS[variant], label=LABELS[variant])
            for x, value, successes, total in zip(AGENTS, values, rows.successes, rows.episodes):
                axis.annotate(
                    f"{value:.0f}% ({int(successes)}/{int(total)})", (x, value),
                    xytext=(0, offsets[variant]), textcoords="offset points", ha="center",
                    fontsize=8, color=COLORS[variant],
                )
        axis.set_title("Random maps" if suite == "random" else "Maze maps")
        axis.set_xlabel("Number of agents")
        axis.set_xticks(AGENTS)
        axis.set_ylim(-4, 110)
    axes[0].set_ylabel("Success Rate, SR (%)")
    axes[1].legend()
    fig.suptitle("MAPF Transformer ablation: rollout success rate", fontsize=15)
    save(fig, args.output_dir / "success_rate_comparison")

    order = [(suite, agents) for suite in SUITES for agents in AGENTS]
    plot_data = paired.set_index(["suite", "num_agents"]).loc[order].reset_index()
    labels = [f"{'Random' if suite == 'random' else 'Maze'}\n{agents} agents" for suite, agents in order]
    x = np.arange(len(plot_data))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("SoC", "makespan", "runtime"),
        ("Sum of Costs (SoC)", "Makespan", "Runtime (seconds)"),
    ):
        maximum = 0.0
        for offset, variant in zip((-width, 0, width), VARIANTS):
            values = plot_data[f"{variant}_{metric}"].to_numpy(float)
            finite = values[np.isfinite(values)]
            if finite.size:
                maximum = max(maximum, float(finite.max()))
            bars = axis.bar(x + offset, values, width, color=COLORS[variant], label=LABELS[variant])
            axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, rotation=90)
        axis.set_title(title)
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_ylim(0, maximum * 1.32 if maximum else 1)
        for index, count in enumerate(plot_data.all_success_episodes):
            axis.text(index, maximum * 1.25, f"n={int(count)}", ha="center", va="top", fontsize=9)
    axes[0].set_ylabel("Mean on episodes solved by all three variants")
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Paired comparison where all three variants achieved SR=1", fontsize=15)
    save(fig, args.output_dir / "all_success_metrics")

    print(sr.to_string(index=False))
    print("\nAll-success comparison")
    print(paired.to_string(index=False))
    print(f"\noutput={args.output_dir}")


if __name__ == "__main__":
    main()
