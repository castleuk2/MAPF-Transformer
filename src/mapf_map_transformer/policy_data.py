from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def crop_halo_map(obstacles: np.ndarray, center_xy: np.ndarray, size: int = 17) -> np.ndarray:
    """Ego-centered static crop with out-of-map cells treated as occupied."""
    x, y = (int(v) for v in center_xy)
    radius = size // 2
    result = np.ones((size, size), dtype=np.uint8)
    x0, y0 = max(0, x - radius), max(0, y - radius)
    x1, y1 = min(obstacles.shape[0], x + radius + 1), min(obstacles.shape[1], y + radius + 1)
    if x0 < x1 and y0 < y1:
        dx, dy = x0 - (x - radius), y0 - (y - radius)
        result[dx : dx + x1 - x0, dy : dy + y1 - y0] = obstacles[x0:x1, y0:y1]
    return result


class _EpisodeLRU:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max_items
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        with np.load(path, allow_pickle=False) as archive:
            value = (
                np.asarray(archive["obstacles"], dtype=np.uint8),
                np.asarray(archive["positions"], dtype=np.int16),
            )
        self.cache[path] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


class PolicyHistoryHaloDataset(Dataset[dict[str, torch.Tensor]]):
    """Rebuild 17x17 halo maps with the baseline policy-history exposure rules."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        history_frames: int = 5,
        history_augmentation: bool = False,
        min_history_frames: int = 1,
        goal_wait_keep_ratio: float = 0.2,
        seed: int = 42,
        cache_size: int = 8,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.history_frames = int(history_frames)
        self.history_augmentation = bool(history_augmentation)
        self.min_history_frames = int(min_history_frames)
        self.goal_wait_keep_ratio = float(goal_wait_keep_ratio)
        self.seed = int(seed)
        self.records: list[dict] = []
        counts: list[int] = []
        for raw_line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            source = Path(record.get("source_path", record["path"]))
            if not source.is_absolute():
                source = self.manifest_path.parent / source
            time_steps = int(record["time_steps"])
            num_agents = int(record["num_agents"])
            arrivals = np.asarray(record.get("arrival_steps", [time_steps] * num_agents), dtype=np.int64)
            suffix = np.maximum(0, time_steps - arrivals)
            retained = np.where(
                suffix > 0,
                np.minimum(suffix, np.maximum(1, np.ceil(suffix * self.goal_wait_keep_ratio).astype(np.int64))),
                0,
            )
            if self.goal_wait_keep_ratio == 0.0:
                retained.fill(0)
            sample_counts = arrivals + retained
            count = int(sample_counts.sum())
            family = str(record.get("map_family", "random")).lower()
            self.records.append({
                **record, "source_path": str(source.resolve()), "count": count,
                "arrivals_array": arrivals, "sample_counts_array": sample_counts,
                "agent_cumulative": np.cumsum(sample_counts),
                "map_family_id": 0 if family == "maze" else 1,
            })
            counts.append(count)
        if not self.records:
            raise ValueError(f"Empty policy-history manifest: {self.manifest_path}")
        self.counts = np.asarray(counts, dtype=np.int64)
        self.cumulative = np.cumsum(self.counts)
        self.cache = _EpisodeLRU(cache_size)

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def resolve_index(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            index += len(self)
        episode_index = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = int(self.cumulative[episode_index - 1]) if episode_index else 0
        local = index - previous
        record = self.records[episode_index]
        ego_id = int(np.searchsorted(record["agent_cumulative"], local, side="right"))
        agent_previous = int(record["agent_cumulative"][ego_id - 1]) if ego_id else 0
        sample_index = int(local - agent_previous)
        arrival = int(record["arrivals_array"][ego_id])
        if sample_index < arrival:
            frame = sample_index
        else:
            suffix_length = max(0, int(record["time_steps"]) - arrival)
            keep = int(record["sample_counts_array"][ego_id]) - arrival
            offsets = np.linspace(0, suffix_length - 1, num=keep, dtype=np.int64)
            frame = arrival + int(offsets[sample_index - arrival])
        return episode_index, frame, ego_id

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, frame, ego_id = self.resolve_index(index)
        record = self.records[episode_index]
        available = min(self.history_frames, frame + 1)
        keep = available
        if self.history_augmentation:
            low = min(self.min_history_frames, available)
            rng = np.random.default_rng(self.seed + index * 104729)
            keep = int(rng.integers(low, available + 1))
        first = frame - keep + 1
        obstacles, positions = self.cache.get(record["source_path"])
        maps = np.ones((self.history_frames, 17, 17), dtype=np.uint8)
        valid = np.zeros(self.history_frames, dtype=bool)
        for slot, source_frame in enumerate(range(first, frame + 1), start=self.history_frames - keep):
            maps[slot] = crop_halo_map(obstacles, positions[source_frame, ego_id])
            valid[slot] = True
        return {
            "halo_maps": torch.from_numpy(maps).long(),
            "frame_valid": torch.from_numpy(valid),
            "map_family": torch.tensor(record["map_family_id"], dtype=torch.long),
            "num_agents": torch.tensor(record["num_agents"], dtype=torch.long),
        }


class EpisodeBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: PolicyHistoryHaloDataset, batch_size: int, shuffle: bool, seed: int) -> None:
        self.dataset, self.batch_size, self.shuffle, self.seed = dataset, int(batch_size), bool(shuffle), int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        episodes = np.arange(len(self.dataset.records))
        if self.shuffle:
            rng.shuffle(episodes)
        starts = np.concatenate(([0], self.dataset.cumulative[:-1]))
        for episode in episodes:
            local = np.arange(int(self.dataset.counts[episode]))
            if self.shuffle:
                rng.shuffle(local)
            absolute = local + int(starts[episode])
            for start in range(0, len(absolute), self.batch_size):
                yield absolute[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(math.ceil(int(count) / self.batch_size) for count in self.dataset.counts)
