from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), smoothed))


def optional_metric(events: list[dict[str, object]], key: str) -> np.ndarray | None:
    if not events or not all(event.get(key) is not None for event in events):
        return None
    return np.asarray([event[key] for event in events], dtype=float)


def unavailable(axis: plt.Axes, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="#666666",
        fontsize=11,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPF Transformer training metrics")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Output path without suffix")
    parser.add_argument("--smooth", type=int, default=25, help="Train-loss moving-average window")
    args = parser.parse_args()

    events = load_events(args.metrics)
    train = [event for event in events if event.get("event") == "train"]
    validation = [event for event in events if event.get("event") == "validation"]
    complete = next((event for event in reversed(events) if event.get("event") == "complete"), {})
    if not train:
        raise ValueError(f"No train events found in {args.metrics}")

    train_steps = np.asarray([event["step"] for event in train], dtype=int)
    train_loss = np.asarray([event["loss"] for event in train], dtype=float)
    train_action_loss = optional_metric(train, "action_loss")
    train_map_loss = optional_metric(train, "map_reconstruction_loss")
    train_weighted_map_loss = optional_metric(train, "weighted_map_reconstruction_loss")
    learning_rate = np.asarray([event["learning_rate"] for event in train], dtype=float)
    throughput = np.asarray([event["samples_per_second"] for event in train], dtype=float)
    val_steps = np.asarray([event["step"] for event in validation], dtype=int)
    val_loss = np.asarray([event["loss"] for event in validation], dtype=float)
    val_accuracy = np.asarray([event["accuracy"] for event in validation], dtype=float)
    val_action_loss = optional_metric(validation, "action_loss")
    val_map_loss = optional_metric(validation, "map_reconstruction_loss")
    val_weighted_map_loss = optional_metric(validation, "weighted_map_reconstruction_loss")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)

    axes[0, 0].plot(train_steps, train_loss, color="#4C78A8", alpha=0.18, linewidth=0.8, label="Train loss")
    axes[0, 0].plot(
        train_steps,
        moving_average(train_loss, max(1, args.smooth)),
        color="#1F4E79",
        linewidth=2,
        label=f"Train loss ({args.smooth}-point mean)",
    )
    if len(validation):
        axes[0, 0].plot(val_steps, val_loss, "o-", color="#E45756", markersize=4, label="Validation loss")
    axes[0, 0].set(title="Total training and validation loss", xlabel="Optimizer step", ylabel="Total loss")
    axes[0, 0].legend()

    if train_action_loss is not None and train_map_loss is not None:
        axes[0, 1].plot(
            train_steps,
            moving_average(train_action_loss, max(1, args.smooth)),
            color="#4C78A8",
            linewidth=2,
            label="Action loss",
        )
        axes[0, 1].plot(
            train_steps,
            moving_average(train_map_loss, max(1, args.smooth)),
            color="#F58518",
            linewidth=2,
            label="Raw map reconstruction loss",
        )
        if train_weighted_map_loss is not None:
            axes[0, 1].plot(
                train_steps,
                moving_average(train_weighted_map_loss, max(1, args.smooth)),
                color="#B279A2",
                linewidth=1.8,
                linestyle="--",
                label="Weighted map loss",
            )
        axes[0, 1].legend()
    else:
        unavailable(axes[0, 1], "Decomposed train losses were not recorded\nby this historical run")
    axes[0, 1].set(
        title=f"Decomposed training losses ({args.smooth}-point mean)",
        xlabel="Optimizer step",
        ylabel="Loss",
    )

    if len(validation) and val_action_loss is not None:
        axes[1, 0].plot(val_steps, val_action_loss, "o-", color="#4C78A8", markersize=4, label="Action loss")
        if val_map_loss is not None:
            axes[1, 0].plot(
                val_steps,
                val_map_loss,
                "o-",
                color="#F58518",
                markersize=4,
                label="Raw map reconstruction loss",
            )
        if val_weighted_map_loss is not None:
            axes[1, 0].plot(
                val_steps,
                val_weighted_map_loss,
                "o--",
                color="#B279A2",
                markersize=4,
                label="Weighted map loss",
            )
        axes[1, 0].legend()
    else:
        unavailable(axes[1, 0], "Decomposed validation losses are unavailable")
    axes[1, 0].set(title="Decomposed validation losses", xlabel="Optimizer step", ylabel="Loss")

    if len(validation):
        axes[1, 1].plot(val_steps, val_accuracy * 100, "o-", color="#54A24B", markersize=4)
        best_index = int(np.argmax(val_accuracy))
        axes[1, 1].scatter(val_steps[best_index], val_accuracy[best_index] * 100, color="#1B6E1B", zorder=3)
        axes[1, 1].annotate(
            f"{val_accuracy[best_index] * 100:.2f}%",
            (val_steps[best_index], val_accuracy[best_index] * 100),
            xytext=(8, -14),
            textcoords="offset points",
        )
    axes[1, 1].set(title="Validation action accuracy", xlabel="Optimizer step", ylabel="Accuracy (%)")

    axes[2, 0].plot(train_steps, learning_rate, color="#B279A2", linewidth=2)
    axes[2, 0].set(title="Learning-rate schedule", xlabel="Optimizer step", ylabel="Learning rate")
    axes[2, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[2, 1].plot(train_steps, throughput, color="#F58518", alpha=0.3, linewidth=0.8)
    axes[2, 1].plot(train_steps, moving_average(throughput, max(1, args.smooth)), color="#B85C00", linewidth=2)
    axes[2, 1].set(title="Training throughput", xlabel="Optimizer step", ylabel="Samples / second")

    final_step = complete.get("step", int(train_steps[-1]))
    final_val_loss = complete.get("validation_loss")
    final_val_accuracy = complete.get("validation_accuracy")
    loss_text = f"{float(final_val_loss):.4f}" if final_val_loss is not None else "in progress"
    accuracy_text = f"{float(final_val_accuracy) * 100:.2f}%" if final_val_accuracy is not None else "n/a"
    fig.suptitle(
        f"MAPF Transformer training summary — step {final_step:,}, "
        f"final val loss {loss_text}, final val accuracy {accuracy_text}",
        fontsize=15,
    )

    output = args.output or args.metrics.with_name("training_metrics")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    print(output.with_suffix(".png"))
    print(output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
