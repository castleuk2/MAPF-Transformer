import numpy as np

from mapf_transformer.config import ModelConfig
from mapf_transformer.dataset import (
    EpisodeData,
    EpisodeSequenceDataset,
    SequenceSampleBuilder,
    save_episode,
    write_manifest,
)
from mapf_transformer.synthetic import generate_synthetic_episode


def test_initial_history_is_padded_without_separate_dataset():
    config = ModelConfig(d_model=32, n_heads=4, temporal_layers=1, map_latents=4)
    episode = generate_synthetic_episode(seed=5, num_agents=3, max_steps=4)
    builder = SequenceSampleBuilder(config)

    first = builder.build(episode, ego_id=0, time_step=0)
    assert first["frame_valid"].sum().item() == 1
    assert first["local_maps"].shape == (
        config.history_frames,
        config.map_size,
        config.map_size,
    )
    assert first["agent_x"].shape == (config.history_frames, config.agents_per_frame)
    assert "action_mask" not in first
    assert first["one_hop_ctg"].shape == (
        config.history_frames,
        config.agents_per_frame,
        5,
    )

    later_time = min(2, episode.time_steps - 1)
    later = builder.build(episode, ego_id=0, time_step=later_time)
    assert later["frame_valid"].sum().item() == later_time + 1


def test_dataset_excludes_only_waits_after_final_goal_arrival(tmp_path):
    goals = np.asarray([[0, 1], [1, 2]], dtype=np.int16)
    positions = np.asarray([
        [[0, 0], [1, 0]],  # t=0
        [[0, 1], [1, 0]],  # agent 0 first reaches goal
        [[0, 0], [1, 1]],  # agent 0 leaves to yield
        [[0, 1], [1, 1]],  # agent 0 finally reaches goal
        [[0, 1], [1, 1]],  # post-goal wait for agent 0
        [[0, 1], [1, 2]],  # agent 1 finally reaches goal
    ], dtype=np.int16)
    episode = EpisodeData(
        obstacles=np.zeros((3, 3), dtype=np.uint8),
        positions=positions,
        goals=goals,
        actions=np.zeros((5, 2), dtype=np.uint8),
    )
    assert episode.get_arrival_steps().tolist() == [3, 5]
    path = tmp_path / "episode.npz"
    save_episode(path, episode)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([path], manifest)

    dataset = EpisodeSequenceDataset(manifest, ModelConfig())
    assert len(dataset) == 8  # SoC, not makespan(5) * agents(2) = 10
    assert dataset._resolve_index(2) == (0, 2, 0)
    assert dataset._resolve_index(3) == (0, 0, 1)
    assert dataset._resolve_index(7) == (0, 4, 1)


def test_dataset_retains_sampled_final_goal_wait_targets(tmp_path):
    goals = np.asarray([[0, 1], [1, 2]], dtype=np.int16)
    positions = np.asarray([
        [[0, 0], [1, 0]],
        [[0, 1], [1, 0]],
        [[0, 0], [1, 1]],
        [[0, 1], [1, 1]],
        [[0, 1], [1, 1]],
        [[0, 1], [1, 2]],
    ], dtype=np.int16)
    actions = np.asarray([
        [4, 0], [3, 4], [4, 0], [0, 0], [0, 4]
    ], dtype=np.uint8)
    episode = EpisodeData(
        obstacles=np.zeros((3, 3), dtype=np.uint8),
        positions=positions,
        goals=goals,
        actions=actions,
    )
    path = tmp_path / "episode.npz"
    save_episode(path, episode)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([path], manifest)

    dataset = EpisodeSequenceDataset(
        manifest, ModelConfig(), goal_wait_keep_ratio=0.2
    )
    assert len(dataset) == 9  # SoC 8 + one sampled stable-goal WAIT.
    assert dataset._resolve_index(3) == (0, 3, 0)
    assert dataset[3]["target"].item() == 0

    all_waits = EpisodeSequenceDataset(
        manifest, ModelConfig(), goal_wait_keep_ratio=1.0
    )
    assert len(all_waits) == 10
    assert all_waits._resolve_index(4) == (0, 4, 0)


def test_builder_prefers_precomputed_24_slot_tracking():
    config = ModelConfig(max_neighbors=24)
    episode = generate_synthetic_episode(seed=17, num_agents=3, max_steps=4)
    episode.positions[:, 1] = episode.positions[:, 0] + np.asarray([0, 1], dtype=np.int16)
    frames = episode.positions.shape[0]
    episode.neighbor_ids_24 = np.full((frames, 3, 24), -1, dtype=np.int16)
    episode.neighbor_valid_24 = np.zeros((frames, 3, 24), dtype=bool)
    episode.track_reset_24 = np.zeros((frames, 3, 24), dtype=bool)
    episode.neighbor_ids_24[:, 0, 7] = 1
    episode.neighbor_valid_24[:, 0, 7] = True
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=0)
    current = -1
    assert sample["agent_valid"][current, 7]
    assert not sample["agent_valid"][current, 0]
