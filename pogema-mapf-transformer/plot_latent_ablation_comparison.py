from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["suite", "map_name", "seed", "num_agents"]
SUITES = ["random", "mazes"]
AGENTS = [8, 16, 24]
COLORS = {"latent16": "#4C78A8", "latent32": "#E45756"}
LABELS = {"latent16": "Map latent 16 (baseline)", "latent32": "Map latent 32"}


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def load(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "model" in frame:
        frame = frame[frame.model == "mapf_transformer"].copy()
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate scenario keys in {path}")
    frame["variant"] = label
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare map-latent 16 and 32 rollouts")
    parser.add_argument("baseline_episodes", type=Path)
    parser.add_argument("latent32_episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load(args.baseline_episodes, "latent16").set_index(KEYS).sort_index()
    latent32 = load(args.latent32_episodes, "latent32").set_index(KEYS).sort_index()
    if not baseline.index.equals(latent32.index):
        raise ValueError(
            f"Scenario mismatch: baseline={len(baseline)}, latent32={len(latent32)}, "
            f"baseline_only={len(baseline.index.difference(latent32.index))}, "
            f"latent32_only={len(latent32.index.difference(baseline.index))}"
        )

    episodes = pd.concat([baseline.reset_index(), latent32.reset_index()], ignore_index=True)
    episodes.to_csv(args.output_dir / "episodes.csv", index=False)
    sr = (
        episodes.groupby(["variant", "suite", "num_agents"], as_index=False)
        .agg(episodes=("CSR", "size"), successes=("CSR", "sum"), SR=("CSR", "mean"), ISR=("ISR", "mean"))
    )
    sr.to_csv(args.output_dir / "sr_by_map_agent.csv", index=False)

    transition_rows = []
    paired_rows = []
    paired_episodes = []
    for suite in SUITES:
        for agents in AGENTS:
            keys = [key for key in baseline.index if key[0] == suite and int(key[3]) == agents]
            base_success = np.asarray([baseline.loc[key, "CSR"] == 1 for key in keys])
            new_success = np.asarray([latent32.loc[key, "CSR"] == 1 for key in keys])
            transition_rows.append(
                {
                    "suite": suite,
                    "num_agents": agents,
                    "both_success": int(np.sum(base_success & new_success)),
                    "baseline_only": int(np.sum(base_success & ~new_success)),
                    "latent32_only": int(np.sum(~base_success & new_success)),
                    "both_failed": int(np.sum(~base_success & ~new_success)),
                }
            )
            common = [key for key, keep in zip(keys, base_success & new_success) if keep]
            row = {"suite": suite, "num_agents": agents, "both_success_episodes": len(common)}
            for variant, frame in (("latent16", baseline), ("latent32", latent32)):
                for metric in ("SoC", "makespan", "runtime"):
                    row[f"{variant}_{metric}"] = (
                        float(np.mean([float(frame.loc[key, metric]) for key in common])) if common else np.nan
                    )
            paired_rows.append(row)
            for key in common:
                item = dict(zip(KEYS, key))
                for variant, frame in (("latent16", baseline), ("latent32", latent32)):
                    for metric in ("SoC", "makespan", "runtime"):
                        item[f"{variant}_{metric}"] = float(frame.loc[key, metric])
                paired_episodes.append(item)

    transitions = pd.DataFrame(transition_rows)
    paired = pd.DataFrame(paired_rows)
    transitions.to_csv(args.output_dir / "success_transitions.csv", index=False)
    paired.to_csv(args.output_dir / "both_success_comparison.csv", index=False)
    pd.DataFrame(paired_episodes).to_csv(args.output_dir / "both_success_episodes.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, constrained_layout=True)
    for axis, suite in zip(axes, SUITES):
        for variant in ("latent16", "latent32"):
            rows = sr[(sr.variant == variant) & (sr.suite == suite)].set_index("num_agents").loc[AGENTS]
            values = rows.SR.to_numpy(float) * 100
            axis.plot(AGENTS, values, "o-", lw=2.5, color=COLORS[variant], label=LABELS[variant])
            for x, value, successes, total in zip(AGENTS, values, rows.successes, rows.episodes):
                offset = 9 if variant == "latent32" else -18
                axis.annotate(f"{value:.0f}% ({int(successes)}/{int(total)})", (x, value),
                              xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9)
        axis.set_title("Random maps" if suite == "random" else "Maze maps")
        axis.set_xlabel("Number of agents")
        axis.set_xticks(AGENTS)
        axis.set_ylim(-3, 105)
    axes[0].set_ylabel("Success Rate, SR (%)")
    axes[1].legend()
    fig.suptitle("Map-latent ablation: rollout success rate", fontsize=15)
    save(fig, args.output_dir / "success_rate_comparison")

    order = [(suite, agents) for suite in SUITES for agents in AGENTS]
    plot_data = paired.set_index(["suite", "num_agents"]).loc[order].reset_index()
    x = np.arange(len(plot_data))
    labels = [f"{'Random' if suite == 'random' else 'Maze'}\n{agents} agents" for suite, agents in order]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for axis, metric, title in zip(axes, ("SoC", "makespan", "runtime"),
                                   ("Sum of Costs (SoC)", "Makespan", "Runtime (seconds)")):
        maximum = 0.0
        for index, variant in enumerate(("latent16", "latent32")):
            values = plot_data[f"{variant}_{metric}"].to_numpy(float)
            maximum = max(maximum, float(np.nanmax(values)))
            bars = axis.bar(x + (-0.18 if index == 0 else 0.18), values, 0.36,
                            color=COLORS[variant], label=LABELS[variant])
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, rotation=90)
        axis.set_title(title)
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_ylim(0, maximum * 1.30)
        for index, count in enumerate(plot_data.both_success_episodes):
            axis.text(index, maximum * 1.24, f"n={int(count)}", ha="center", va="top", fontsize=9)
    axes[0].set_ylabel("Mean on episodes solved by both variants")
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Paired path-quality comparison (both SR=1)", fontsize=15)
    save(fig, args.output_dir / "both_success_metrics")

    print(sr.to_string(index=False))
    print("\nSuccess transitions")
    print(transitions.to_string(index=False))
    print(f"\noutput={args.output_dir}")


if __name__ == "__main__":
    main()
