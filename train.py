from __future__ import annotations

import argparse

from mapf_map_transformer.config import load_experiment_config
from mapf_map_transformer.trainer import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 17x17-to-25-token structured map autoencoder.")
    parser.add_argument("--config", required=True, help="YAML experiment configuration.")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional smoke-test step limit.")
    parser.add_argument("--device", default=None, help="Override training.device, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.device is not None:
        config.training.device = args.device
    result = train_experiment(config)
    print(f"best_checkpoint={result.best_checkpoint}")
    print(f"last_checkpoint={result.last_checkpoint}")


if __name__ == "__main__":
    main()
