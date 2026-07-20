from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import cppimport.import_hook  # noqa: F401  # builds the official C++ tokenizer modules


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from tokenizer import cost2go  # noqa: E402
from tokenizer.encoder import AgentsInfo, Encoder, InputParameters as CppInputParameters  # noqa: E402
from tokenizer.parameters import InputParameters  # noqa: E402


MOVES_TO_TOKEN = {
    (0, 0): "w",
    (-1, 0): "u",
    (1, 0): "d",
    (0, -1): "l",
    (0, 1): "r",
}

# Each process reuses static cost-to-go fields across seeds/agent counts of the
# same map. Dataset manifests are map-major, so this avoids most recomputation.
_COST2GO_CACHE: dict[str, Any] = {}
_COST2GO_CACHE_ORDER: list[str] = []
_COST2GO_CACHE_SIZE = 8


def cached_cost2go(obstacles: np.ndarray, radius: int) -> Any:
    key = hashlib.sha256(obstacles.tobytes()).hexdigest()
    if key in _COST2GO_CACHE:
        _COST2GO_CACHE_ORDER.remove(key)
        _COST2GO_CACHE_ORDER.append(key)
        return _COST2GO_CACHE[key]
    fields = cost2go.precompute_cost2go(obstacles.tolist(), radius)
    _COST2GO_CACHE[key] = fields
    _COST2GO_CACHE_ORDER.append(key)
    while len(_COST2GO_CACHE_ORDER) > _COST2GO_CACHE_SIZE:
        oldest = _COST2GO_CACHE_ORDER.pop(0)
        del _COST2GO_CACHE[oldest]
    return fields


@dataclass(frozen=True)
class ConversionConfig:
    goal_wait_keep_ratio: float = 0.2
    shard_size: int = 65_536
    num_agents: int = 13
    num_previous_actions: int = 5
    agents_radius: int = 5
    cost2go_value_limit: int = 20
    cost2go_radius: int = 5
    context_size: int = 256


def read_manifest(path: Path, limit_episodes: int | None = None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit_episodes is not None:
        records = records[:limit_episodes]
    for record in records:
        episode_path = Path(record["path"])
        if not episode_path.is_absolute():
            episode_path = path.parent / episode_path
        record["path"] = str(episode_path.resolve())
    return records


def retained_goal_wait_steps(arrival: int, time_steps: int, ratio: float) -> np.ndarray:
    suffix_length = max(0, int(time_steps) - int(arrival))
    if suffix_length == 0 or ratio <= 0:
        return np.empty(0, dtype=np.int64)
    keep = min(suffix_length, max(1, int(math.ceil(suffix_length * ratio))))
    offsets = np.linspace(0, suffix_length - 1, num=keep, dtype=np.int64)
    return int(arrival) + offsets


def previous_action_tokens(path: np.ndarray, time: int, count: int) -> list[str]:
    start = max(1, time - count + 1)
    tokens = [
        MOVES_TO_TOKEN[tuple((path[index] - path[index - 1]).tolist())]
        for index in range(start, time + 1)
    ]
    if time < count:
        tokens = ["n"] * (count - time) + tokens
    if len(tokens) < count:
        tokens += ["w"] * (count - len(tokens))
    return tokens[-count:]


def agent_info(
    agent_id: int,
    ego_position: np.ndarray,
    time_step: int,
    paths: np.ndarray,
    goals: np.ndarray,
    distance_fields: dict[tuple[int, int], Any],
    previous_actions: int,
) -> AgentsInfo:
    position = paths[time_step, agent_id]
    goal = goals[agent_id]
    relative_position = tuple(int(v) for v in position - ego_position)
    relative_goal = tuple(int(v) for v in goal - ego_position)
    field = distance_fields[tuple(int(v) for v in goal)]
    current = int(field[int(position[0])][int(position[1])])
    next_action = ""
    for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = int(position[0] + delta[0]), int(position[1] + delta[1])
        value = int(field[nx][ny])
        next_action += "1" if value >= 0 and current > value else "0"
    return AgentsInfo(
        relative_position,
        relative_goal,
        previous_action_tokens(paths[:, agent_id], time_step, previous_actions),
        next_action,
    )


def convert_episode(task: tuple[dict[str, Any], ConversionConfig]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    record, config = task
    with np.load(record["path"], allow_pickle=False) as data:
        obstacles = np.asarray(data["obstacles"], dtype=np.uint8)
        positions = np.asarray(data["positions"], dtype=np.int16)
        goals_raw = np.asarray(data["goals"], dtype=np.int16)
        goals = goals_raw if goals_raw.ndim == 2 else goals_raw[-1]
        actions = np.asarray(data["actions"], dtype=np.int8)
        arrivals = np.asarray(data["arrival_steps"], dtype=np.int64)

    parameters = CppInputParameters(
        config.cost2go_value_limit,
        config.num_agents,
        config.num_previous_actions,
        config.context_size,
    )
    encoder = Encoder(parameters)
    fields = cached_cost2go(obstacles, config.cost2go_radius)
    selected: list[tuple[int, int]] = []
    for ego_id, arrival in enumerate(arrivals.tolist()):
        selected.extend((ego_id, t) for t in range(int(arrival)))
        selected.extend(
            (ego_id, int(t))
            for t in retained_goal_wait_steps(int(arrival), actions.shape[0], config.goal_wait_keep_ratio)
        )

    inputs = np.empty((len(selected), config.context_size), dtype=np.int8)
    targets = np.empty(len(selected), dtype=np.int8)
    radius = config.agents_radius
    for output_index, (ego_id, time_step) in enumerate(selected):
        ego_position = positions[time_step, ego_id]
        ego_field = fields[tuple(int(v) for v in ego_position)]
        candidates: list[tuple[int, int]] = []
        for agent_id, position in enumerate(positions[time_step]):
            delta = np.abs(position - ego_position)
            if int(delta[0]) <= radius and int(delta[1]) <= radius:
                distance = int(ego_field[int(position[0])][int(position[1])])
                if distance >= 0:
                    candidates.append((agent_id, distance))
        candidates.sort(key=lambda item: (item[1], item[0]))
        agents = [
            agent_info(
                agent_id,
                ego_position,
                time_step,
                positions,
                goals,
                fields,
                config.num_previous_actions,
            )
            for agent_id, _ in candidates[: config.num_agents]
        ]
        goal_field = fields[tuple(int(v) for v in goals[ego_id])]
        cost_observation = cost2go.generate_cost2go_obs(
            goal_field,
            tuple(int(v) for v in ego_position),
            config.cost2go_radius,
            config.cost2go_value_limit,
            False,
        )
        inputs[output_index] = np.asarray(encoder.encode(agents, cost_observation), dtype=np.int8)
        targets[output_index] = actions[time_step, ego_id]

    metadata = {
        "path": record["path"],
        "samples": len(selected),
        "soc_samples": int(arrivals.sum()),
        "goal_wait_samples": len(selected) - int(arrivals.sum()),
    }
    return inputs, targets, metadata


class ArrowShardWriter:
    def __init__(self, output_dir: Path, shard_size: int, context_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.context_size = context_size
        self.inputs: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.buffered = 0
        self.shard_index = 0
        self.total = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        offset = 0
        while offset < len(inputs):
            take = min(self.shard_size - self.buffered, len(inputs) - offset)
            self.inputs.append(inputs[offset : offset + take])
            self.targets.append(targets[offset : offset + take])
            self.buffered += take
            offset += take
            if self.buffered == self.shard_size:
                self.flush()

    def flush(self) -> None:
        if self.buffered == 0:
            return
        inputs = np.concatenate(self.inputs)
        targets = np.concatenate(self.targets)
        values = pa.array(inputs.reshape(-1), type=pa.int8())
        input_column = pa.FixedSizeListArray.from_arrays(values, self.context_size)
        table = pa.Table.from_arrays(
            [input_column, pa.array(targets, type=pa.int8())],
            names=["input_tensors", "gt_actions"],
        )
        path = self.output_dir / f"part_{self.shard_index:05d}.arrow"
        with path.open("wb") as stream:
            with ipc.new_file(stream, table.schema) as writer:
                writer.write(table)
        self.total += len(inputs)
        print(f"wrote {path} samples={len(inputs)} total={self.total}", flush=True)
        self.shard_index += 1
        self.inputs.clear()
        self.targets.clear()
        self.buffered = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert existing MAPF-LNS2 NPZ trajectories to MAPF-GPT Arrow")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--shard-size", type=int, default=65_536)
    parser.add_argument("--goal-wait-keep-ratio", type=float, default=0.2)
    parser.add_argument("--limit-episodes", type=int, default=None)
    args = parser.parse_args()
    if args.workers <= 0 or args.shard_size <= 0:
        parser.error("workers and shard-size must be positive")
    if not 0 <= args.goal_wait_keep_ratio <= 1:
        parser.error("goal-wait-keep-ratio must be in [0,1]")

    config = ConversionConfig(
        goal_wait_keep_ratio=args.goal_wait_keep_ratio,
        shard_size=args.shard_size,
    )
    records = read_manifest(args.manifest.resolve(), args.limit_episodes)
    writer = ArrowShardWriter(args.output_dir.resolve(), config.shard_size, config.context_size)
    episode_metadata: list[dict[str, Any]] = []
    tasks = [(record, config) for record in records]
    if args.workers == 1:
        results = map(convert_episode, tasks)
        for inputs, targets, metadata in results:
            writer.add(inputs, targets)
            episode_metadata.append(metadata)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for inputs, targets, metadata in executor.map(convert_episode, tasks, chunksize=1):
                writer.add(inputs, targets)
                episode_metadata.append(metadata)
    writer.flush()

    summary = {
        "manifest": str(args.manifest.resolve()),
        "episodes": len(records),
        "samples": writer.total,
        "soc_samples": sum(item["soc_samples"] for item in episode_metadata),
        "goal_wait_samples": sum(item["goal_wait_samples"] for item in episode_metadata),
        "shards": writer.shard_index,
        "config": asdict(config),
    }
    (args.output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
