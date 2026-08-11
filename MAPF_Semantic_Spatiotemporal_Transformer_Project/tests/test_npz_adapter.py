from pathlib import Path

import numpy as np
import torch

from mapf_sst.data.npz_dataset import EpisodeFeatureBuilder
from mapf_sst.cpp import CppEpisodeFeatureBuilder
from dataclasses import fields


def test_npz_episode_adapter_and_all_views(cfg, tmp_path: Path):
    obstacles = np.zeros((20, 20), dtype=np.uint8)
    obstacles[5, 5:10] = 1
    positions = np.array(
        [
            [[10, 10], [10, 12], [12, 10]],
            [[10, 11], [10, 13], [11, 10]],
            [[10, 12], [10, 14], [10, 10]],
        ],
        dtype=np.int64,
    )
    goals = np.array([[10, 16], [10, 17], [10, 8]], dtype=np.int64)
    actions = np.array([[4, 4, 1], [4, 4, 1]], dtype=np.int64)
    path = tmp_path / "episode.npz"
    np.savez(path, obstacles=obstacles, positions=positions, goals=goals, actions=actions)
    builder = EpisodeFeatureBuilder(cfg)
    episode = builder.load_episode(path)
    sample = builder.build(episode, 1, 0)
    assert sample.local_maps.shape == (17, 17)
    assert sample.current_xy.shape == (14, 2)
    assert sample.history_xy.shape == (7, 4, 2)
    batch, graph = builder.build_all_views(episode, 1)
    assert batch.local_maps.shape[0] == 3
    assert graph.neighbor_view_index.shape == (3, 6)
    assert torch.equal(batch.current_global_ids[:, 0], torch.arange(3))

    cpp = CppEpisodeFeatureBuilder(cfg)
    cpp_episode = cpp.load_episode(path)
    cpp_batch, cpp_graph = cpp.build_all_views(cpp_episode, 1)
    for field in fields(batch):
        expected, actual = getattr(batch, field.name), getattr(cpp_batch, field.name)
        if expected is not None:
            assert torch.equal(expected, actual), field.name
    assert torch.equal(graph.neighbor_view_index, cpp_graph.neighbor_view_index)
    assert torch.equal(graph.neighbor_valid, cpp_graph.neighbor_valid)
    assert torch.equal(graph.neighbor_current_slot, cpp_graph.neighbor_current_slot)
