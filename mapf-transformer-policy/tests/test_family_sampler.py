from __future__ import annotations

import numpy as np

from mapf_transformer.family_sampler import FamilyRatioSampler, record_family


class FakeEpisodeDataset:
    def __init__(self) -> None:
        self.records = [
            {"map_family": "maze"},
            {"source_path": "random/train/episode_000.npz"},
        ]
        self.cumulative = np.asarray([6000, 10000], dtype=np.int64)

    def __len__(self) -> int:
        return 10000


def test_record_family_supports_metadata_and_legacy_source_paths():
    assert record_family({"map_family": "maze"}) == "maze"
    assert record_family({"source_path": "maze/train/example.npz"}) == "maze"
    assert record_family({"path": "/cache/random/train/example.npz"}) == "random"


def test_family_sampler_preserves_epoch_length_and_requested_ratio():
    dataset = FakeEpisodeDataset()
    sampler = FamilyRatioSampler(dataset, maze_ratio=0.9, seed=7, chunk_size=257)
    indices = np.asarray(list(sampler), dtype=np.int64)
    assert len(indices) == len(dataset)
    assert np.all((0 <= indices) & (indices < len(dataset)))
    maze_fraction = float(np.mean(indices < 6000))
    assert maze_fraction == 0.9


def test_family_sampler_ddp_rank_lengths_and_epoch_seed():
    dataset = FakeEpisodeDataset()
    rank0 = FamilyRatioSampler(dataset, 0.9, num_replicas=2, rank=0, seed=11)
    rank1 = FamilyRatioSampler(dataset, 0.9, num_replicas=2, rank=1, seed=11)
    assert len(rank0) == len(rank1) == 5000
    first = list(rank0)
    rank0.set_epoch(1)
    assert first != list(rank0)
    assert first != list(rank1)
