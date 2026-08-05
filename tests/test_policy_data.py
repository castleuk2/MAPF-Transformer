from __future__ import annotations

import json

import numpy as np

from mapf_map_transformer.policy_data import PolicyHistoryHaloDataset, crop_halo_map


def test_crop_halo_map_pads_and_centers() -> None:
    obstacles = np.zeros((5, 6), dtype=np.uint8)
    result = crop_halo_map(obstacles, np.array([0, 0]), size=17)
    assert result.shape == (17, 17)
    assert result[8, 8] == 0
    assert result[0, 0] == 1


def test_policy_history_dataset(tmp_path) -> None:
    episode = tmp_path / "episode.npz"
    obstacles = np.zeros((25, 25), dtype=np.uint8)
    positions = np.array([[[12, 12]], [[12, 13]], [[12, 14]]], dtype=np.int16)
    np.savez(episode, obstacles=obstacles, positions=positions)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "path": str(episode), "time_steps": 2, "num_agents": 1,
        "arrival_steps": [2], "map_family": "maze",
    }) + "\n", encoding="utf-8")
    dataset = PolicyHistoryHaloDataset(manifest, history_frames=5, history_augmentation=False)
    item = dataset[1]
    assert item["frame_valid"].tolist() == [False, False, False, True, True]
    assert item["halo_maps"].shape == (5, 17, 17)
    assert item["map_family"].item() == 0
