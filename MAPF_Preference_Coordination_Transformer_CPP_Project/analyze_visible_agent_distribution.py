from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from mapf_pct.config import load_config
from train import build_dataset


def selected_steps(arrival: int, time_steps: int, ratio: float) -> np.ndarray:
    prefix = np.arange(arrival, dtype=np.int64)
    suffix = max(0, time_steps - arrival)
    if suffix == 0 or ratio == 0:
        return prefix
    keep = min(suffix, max(1, int(math.ceil(suffix * ratio))))
    waits = arrival + np.linspace(0, suffix - 1, num=keep, dtype=np.int64)
    return np.concatenate((prefix, waits))


def summarize(counter: Counter[int]) -> dict:
    total = sum(counter.values())
    values = np.fromiter(counter.keys(), dtype=np.int64)
    weights = np.fromiter(counter.values(), dtype=np.int64)
    order = np.argsort(values); values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    def percentile(q: float) -> int:
        return int(values[np.searchsorted(cumulative, q * total, side="left")])
    return {
        "samples": total,
        "mean_including_ego": float((values * weights).sum() / total),
        "mean_neighbors": float((values * weights).sum() / total - 1),
        "median_including_ego": percentile(0.5),
        "p90_including_ego": percentile(0.9),
        "min_including_ego": int(values.min()),
        "max_including_ego": int(values.max()),
        "counts_including_ego": {str(int(k)): int(v) for k, v in sorted(counter.items())},
        "percent_including_ego": {str(int(k)): 100.0 * int(v) / total for k, v in sorted(counter.items())},
    }


def analyze_manifest(manifest: Path, ratio: float, selected_indices: set[int] | None = None):
    overall = Counter(); groups: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    global_index = 0
    for line in manifest.read_text().splitlines():
        if not line.strip(): continue
        record = json.loads(line); positions = np.load(record["path"], allow_pickle=False)["positions"]
        time_steps = int(record["time_steps"]); family = record["map_family"]; n_agents = int(record["num_agents"])
        for ego, arrival in enumerate(record["arrival_steps"]):
            steps = selected_steps(int(arrival), time_steps, ratio)
            for step in steps:
                use = selected_indices is None or global_index in selected_indices
                if use:
                    delta = np.abs(positions[int(step)] - positions[int(step), ego])
                    visible = int((delta.max(axis=1) <= 7).sum())
                    overall[visible] += 1; groups[(family, n_agents)][visible] += 1
                global_index += 1
    return overall, groups, global_index


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_dataset_frozen_map.yaml")
parser.add_argument("--output", type=Path, default=ROOT / "reports/visible_agent_distribution")
args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
config = load_config(args.config); ratio = config.data.goal_wait_keep_ratio
train_manifest = ROOT / config.data.train_manifest; val_manifest = ROOT / config.data.val_manifest
balanced_val = build_dataset(config, False)
balanced_indices = set(map(int, balanced_val.indices))

train, train_groups, train_seen = analyze_manifest(train_manifest, ratio)
full_val, full_val_groups, full_val_seen = analyze_manifest(val_manifest, ratio)
balanced_val_counter, balanced_val_groups, _ = analyze_manifest(val_manifest, ratio, balanced_indices)

payload = {
    "definition": {"core_size": 15, "radius": 7, "distance": "Chebyshev", "includes_ego": True},
    "train": {"overall": summarize(train), "groups": {f"{k[0]}_n{k[1]}": summarize(v) for k, v in sorted(train_groups.items())}},
    "balanced_val_65536": {"overall": summarize(balanced_val_counter), "groups": {f"{k[0]}_n{k[1]}": summarize(v) for k, v in sorted(balanced_val_groups.items())}},
    "full_val_reference": {"overall": summarize(full_val), "groups": {f"{k[0]}_n{k[1]}": summarize(v) for k, v in sorted(full_val_groups.items())}},
}
(args.output / "visible_agent_distribution.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 10})
colors = {"Train": "#2563EB", "Balanced Val": "#EF4444"}

def percentages(counter: Counter[int], x: np.ndarray) -> np.ndarray:
    total = sum(counter.values()); return np.asarray([100 * counter[int(i)] / total for i in x])

maximum = max(max(train), max(balanced_val_counter)); x = np.arange(1, maximum + 1)
fig, ax = plt.subplots(figsize=(12.8, 6.2))
for label, counter in (("Train", train), ("Balanced Val", balanced_val_counter)):
    ax.plot(x, percentages(counter, x), marker="o", markersize=3.5, linewidth=2, label=label, color=colors[label])
ax.set_title("Ego 기준 15×15 내부 Agent 수 분포", fontsize=15, weight="bold")
ax.set_xlabel("15×15 내부 Agent 수 (Ego 포함)"); ax.set_ylabel("Sample 비율 (%)")
ax.set_xticks(x); ax.grid(alpha=.25); ax.legend(frameon=False)
fig.tight_layout()
for suffix in ("png", "pdf"): fig.savefig(args.output / f"01_overall_distribution.{suffix}", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
for ax, key in zip(axes.flat, [(f,n) for f in ("maze","random") for n in (16,24,32)]):
    xmax = max(max(train_groups[key]), max(balanced_val_groups[key])); gx=np.arange(1,xmax+1)
    for label, source in (("Train",train_groups[key]),("Balanced Val",balanced_val_groups[key])):
        ax.plot(gx, percentages(source,gx), marker="o", markersize=3, linewidth=1.8, label=label, color=colors[label])
    ax.set_title(f"{key[0].capitalize()} - {key[1]} Agents", weight="bold")
    ax.set_xlabel("Agent 수 (Ego 포함)"); ax.set_ylabel("비율 (%)"); ax.set_xticks(gx); ax.grid(alpha=.22)
handles, labels = axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="lower center",frameon=False)
fig.suptitle("Map 유형·전체 Agent 수별 15×15 가시 Agent 분포",fontsize=16,weight="bold")
fig.tight_layout(rect=(0,.04,1,.97))
for suffix in ("png", "pdf"): fig.savefig(args.output / f"02_group_distribution.{suffix}", dpi=180, bbox_inches="tight")
plt.close(fig)
print(args.output)
