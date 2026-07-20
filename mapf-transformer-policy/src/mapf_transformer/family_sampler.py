from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


def record_family(record: dict[str, object]) -> str:
    explicit = str(record.get("map_family", "")).lower()
    if explicit in {"maze", "random"}:
        return explicit
    candidate = "/".join(
        str(record.get(key, "")).replace("\\", "/").lower()
        for key in ("source_path", "path", "split")
    )
    components = {component for component in candidate.split("/") if component}
    if "maze" in components or any(component.startswith("maze_") for component in components):
        return "maze"
    if "random" in components or any(component.startswith("random_") for component in components):
        return "random"
    raise ValueError(
        "Cannot infer map family from manifest record; expected map_family or "
        f"a maze/random source path, received {record}"
    )


class FamilyRatioSampler(Sampler[int]):
    """Memory-bounded sample-level Maze/Random balancing for DDP.

    Sampling with replacement is intentional: requesting 90% Maze from a
    corpus whose natural Maze share is lower necessarily reuses Maze samples.
    The logical epoch length remains equal to the original dataset length.
    """

    def __init__(
        self,
        dataset: object,
        maze_ratio: float,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        chunk_size: int = 65_536,
    ) -> None:
        if not 0.0 < maze_ratio < 1.0:
            raise ValueError("maze_ratio must be strictly between 0 and 1")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid num_replicas/rank")
        records = getattr(dataset, "records", None)
        cumulative = np.asarray(getattr(dataset, "cumulative", []), dtype=np.int64)
        if records is None or len(records) != len(cumulative):
            raise TypeError("FamilyRatioSampler requires an EpisodeSequenceDataset")

        starts = np.concatenate((np.zeros(1, dtype=np.int64), cumulative[:-1]))
        counts = cumulative - starts
        self._families: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for family in ("maze", "random"):
            selected = np.asarray(
                [index for index, record in enumerate(records) if record_family(record) == family],
                dtype=np.int64,
            )
            if not len(selected):
                raise ValueError(f"No {family} samples found in training manifest")
            family_counts = counts[selected]
            family_cumulative = np.cumsum(family_counts, dtype=np.int64)
            self._families[family] = (
                starts[selected],
                family_cumulative,
                int(family_cumulative[-1]),
            )

        self.maze_ratio = float(maze_ratio)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_indices(self, family: str, offsets: np.ndarray) -> np.ndarray:
        starts, cumulative, _ = self._families[family]
        episodes = np.searchsorted(cumulative, offsets, side="right")
        previous = np.where(episodes > 0, cumulative[np.maximum(episodes - 1, 0)], 0)
        return starts[episodes] + offsets - previous

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + self.rank * 10_007)
        remaining = self.num_samples
        remaining_maze = int(round(self.num_samples * self.maze_ratio))
        while remaining:
            size = min(self.chunk_size, remaining)
            maze_count = int(round(size * remaining_maze / remaining))
            maze_count = min(size, remaining_maze, maze_count)
            choose_maze = np.zeros(size, dtype=bool)
            choose_maze[:maze_count] = True
            rng.shuffle(choose_maze)
            output = np.empty(size, dtype=np.int64)
            for family, selection in (("maze", choose_maze), ("random", ~choose_maze)):
                count = int(selection.sum())
                if count:
                    total = self._families[family][2]
                    offsets = rng.integers(0, total, size=count, dtype=np.int64)
                    output[selection] = self._global_indices(family, offsets)
            # Avoid long runs of a single family while retaining the ratio.
            rng.shuffle(output)
            yield from output.tolist()
            remaining -= size
            remaining_maze -= maze_count
