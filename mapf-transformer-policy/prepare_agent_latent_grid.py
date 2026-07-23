from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize one config from the 3x3 latent grid")
    parser.add_argument("--grid-config", required=True)
    parser.add_argument("--map-latents", type=int, choices=(32, 48, 64), required=True)
    parser.add_argument("--agent-latents", type=int, choices=(32, 48, 64), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_path = Path(args.grid_config)
    with grid_path.open("r", encoding="utf-8") as stream:
        grid = yaml.safe_load(stream)

    settings = {
        int(item["agent_latents"]): int(item["history_frames"])
        for item in grid["grid"]["agent_settings"]
    }
    if args.map_latents not in {int(value) for value in grid["grid"]["map_latents"]}:
        raise ValueError("Requested map latent count is not present in the grid")

    run_name = f"raw_map{args.map_latents}_agent{args.agent_latents}"
    resolved = {
        "model": {
            **grid["base_model"],
            "map_latents": args.map_latents,
            "agent_latents": args.agent_latents,
            "history_frames": settings[args.agent_latents],
        },
        "training": {
            **grid["base_training"],
            "output_dir": f"mapf-transformer-policy/runs/{run_name}",
        },
    }
    output_dir = Path(resolved["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "launch_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)

    print(
        f"run={run_name} map={args.map_latents} agent={args.agent_latents} "
        f"history={settings[args.agent_latents]} "
        f"context={settings[args.agent_latents] * (args.agent_latents + 1) + 1}"
    )
    print(resolved_path)


if __name__ == "__main__":
    main()
