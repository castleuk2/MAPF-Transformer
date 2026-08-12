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
            [[10, 11], [10, 12], [11, 10]],
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
    assert sample.history_xy.shape == (14, 2, 2)
    # Ego first, then Manhattan-nearest agents (global id breaks ties).
    assert sample.current_global_ids[:3].tolist() == [0, 1, 2]
    assert sample.history_global_ids[:3].tolist() == [0, 1, 2]
    # Agent 1 currently occupies Ego's RIGHT target; dynamic occupancy is
    # computed from every frame agent rather than only static obstacles.
    assert bool(sample.candidate_dynamic_occupied[0, 4])
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


def test_nearest_agent_selection_and_history_prefix(cfg, tmp_path: Path):
    obstacles = np.zeros((30, 30), dtype=np.uint8)
    # ids 1 and 2 tie at distance 1; id order is the deterministic tie-break.
    frame = [[15, 15], [15, 16], [16, 15], [15, 18], [14, 13]]
    positions = np.array([frame, frame], dtype=np.int64)
    goals = positions[0].copy()
    actions = np.zeros((1, 5), dtype=np.int64)
    path = tmp_path / "nearest.npz"
    np.savez(path, obstacles=obstacles, positions=positions, goals=goals, actions=actions)

    builder = EpisodeFeatureBuilder(cfg)
    episode = builder.load_episode(path)
    current = builder._rank_current(episode, 0, 0)
    history = builder._history_ids(episode, 0, 0, current)
    assert current == [0, 1, 2, 3, 4]
    assert history == current[: cfg.history_tracks]
    assert cfg.history_tracks == cfg.max_current_agents == 14
    assert cfg.history_steps == 2
