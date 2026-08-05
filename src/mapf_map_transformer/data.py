from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig, ExperimentConfig
from .synthetic import generate_halo_map
from .policy_data import EpisodeBatchSampler, PolicyHistoryHaloDataset


class SyntheticHaloMapDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        num_samples: int,
        seed: int,
        min_density: float,
        max_density: float,
        pattern_mix: tuple[str, ...],
        ensure_center_free: bool,
    ) -> None:
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.min_density = float(min_density)
        self.max_density = float(max_density)
        self.pattern_mix = tuple(pattern_mix)
        self.ensure_center_free = bool(ensure_center_free)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not self.pattern_mix:
            raise ValueError("pattern_mix must not be empty.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        pattern = str(rng.choice(self.pattern_mix))
        halo_map = generate_halo_map(
            rng,
            pattern=pattern,
            min_density=self.min_density,
            max_density=self.max_density,
            ensure_center_free=self.ensure_center_free,
        )
        # Rotation/reflection augmentation is deterministic for each sample.
        rotations = int(rng.integers(0, 4))
        halo_map = np.rot90(halo_map, rotations)
        if rng.random() < 0.5:
            halo_map = np.fliplr(halo_map)
        halo_map = np.ascontiguousarray(halo_map)
        return {
            "halo_map": torch.from_numpy(halo_map).to(torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class NpzHaloMapDataset(Dataset[dict[str, torch.Tensor]]):
    """Load one or more .npz files containing `halo_maps: [N,17,17]`."""

    def __init__(self, path: str | Path) -> None:
        self.files = self._resolve_files(Path(path))
        self.arrays: list[np.ndarray] = []
        self.offsets = [0]
        for file_path in self.files:
            archive = np.load(file_path, mmap_mode="r")
            key = "halo_maps" if "halo_maps" in archive else "maps" if "maps" in archive else None
            if key is None:
                raise KeyError(f"{file_path} must contain 'halo_maps' or 'maps'.")
            array = archive[key]
            if array.ndim == 2:
                array = array[None, ...]
            if array.ndim != 3 or array.shape[-2:] != (17, 17):
                raise ValueError(f"{file_path}: expected [N,17,17], got {array.shape}")
            self.arrays.append(array)
            self.offsets.append(self.offsets[-1] + len(array))

    @staticmethod
    def _resolve_files(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if path.is_dir():
            files = sorted(path.glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"No .npz files found in {path}")
            return files
        raise FileNotFoundError(path)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local_index = index - self.offsets[file_index]
        halo_map = np.asarray(self.arrays[file_index][local_index], dtype=np.int64)
        return {
            "halo_map": torch.from_numpy(np.array(halo_map, copy=True)).to(torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def build_datasets(config: DatasetConfig) -> tuple[Dataset[Any], Dataset[Any]]:
    if config.kind == "synthetic":
        train_dataset = SyntheticHaloMapDataset(
            num_samples=config.train_samples,
            seed=config.seed,
            min_density=config.min_density,
            max_density=config.max_density,
            pattern_mix=tuple(config.pattern_mix),
            ensure_center_free=config.ensure_center_free,
        )
        val_dataset = SyntheticHaloMapDataset(
            num_samples=config.val_samples,
            seed=config.seed + 1_000_003,
            min_density=config.min_density,
            max_density=config.max_density,
            pattern_mix=tuple(config.pattern_mix),
            ensure_center_free=config.ensure_center_free,
        )
        return train_dataset, val_dataset
    if config.kind == "npz":
        if not config.train_path or not config.val_path:
            raise ValueError("dataset.kind=npz requires train_path and val_path.")
        return NpzHaloMapDataset(config.train_path), NpzHaloMapDataset(config.val_path)
    if config.kind == "policy_history":
        if not config.train_path or not config.val_path:
            raise ValueError("dataset.kind=policy_history requires train_path and val_path.")
        common = dict(
            history_frames=config.history_frames,
            min_history_frames=config.min_history_frames,
            goal_wait_keep_ratio=config.goal_wait_keep_ratio,
        )
        return (
            PolicyHistoryHaloDataset(config.train_path, history_augmentation=config.train_history_augmentation,
                                     seed=config.seed, **common),
            PolicyHistoryHaloDataset(config.val_path, history_augmentation=False,
                                     seed=config.seed + 1, **common),
        )
    raise ValueError(f"Unsupported dataset kind: {config.kind}")


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_dataset, val_dataset = build_datasets(config.dataset)
    if config.dataset.kind == "policy_history":
        train_sampler = EpisodeBatchSampler(train_dataset, config.training.batch_size, True, config.training.seed)
        val_sampler = EpisodeBatchSampler(val_dataset, config.training.val_batch_size, False, config.training.seed + 1)
        kwargs = dict(num_workers=config.training.num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=config.training.num_workers > 0)
        return (DataLoader(train_dataset, batch_sampler=train_sampler, **kwargs),
                DataLoader(val_dataset, batch_sampler=val_sampler, **kwargs))
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
        drop_last=False,
    )
    return train_loader, val_loader
