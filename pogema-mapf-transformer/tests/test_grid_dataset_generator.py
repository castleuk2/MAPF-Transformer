import pytest
import numpy as np

from pogema_mapf_transformer.grid_dataset_generator import _dataset_summary, _select_map_names
from pogema_mapf_transformer.episode_io import EpisodeData


CATALOG = {
    "training-random-seed-00640": "map-a",
    "training-random-seed-00641": "map-b",
    "training-random-seed-00642": "map-c",
}


def test_select_maps_by_catalog_index_range():
    assert _select_map_names(CATALOG, {"map_indices": {"start": 1, "stop": 3}}) == [
        "training-random-seed-00641",
        "training-random-seed-00642",
    ]


def test_select_maps_by_trailing_id_range():
    assert _select_map_names(CATALOG, {"map_ids": {"start": 640, "stop": 642}}) == [
        "training-random-seed-00640",
        "training-random-seed-00641",
    ]


def test_reject_multiple_map_selectors():
    with pytest.raises(ValueError, match="only one map selector"):
        _select_map_names(CATALOG, {"maps": [], "map_indices": [0]})


def test_dataset_summary_counts_ego_agent_actions():
    rows = [
        {"split": "maze_train", "family": "maze", "subset": "train",
         "status": "generated", "num_samples": 320},
        {"split": "random_train", "family": "random", "subset": "train",
         "status": "failed", "num_samples": 0},
        {"split": "maze_val", "family": "maze", "subset": "val",
         "status": "generated", "num_samples": 32},
    ]
    summary = _dataset_summary(rows)
    assert summary["totals"] == {
        "attempted_episodes": 3,
        "generated_episodes": 2,
        "failed_episodes": 1,
        "rectangular_actions": 0,
        "goal_waits_removed": 0,
        "training_samples": 352,
    }
    assert summary["subsets"]["train"]["training_samples"] == 320
    assert summary["train_val_sample_ratio"] == 10.0


def test_arrival_steps_use_final_goal_arrival_not_first_visit():
    episode = EpisodeData(
        obstacles=np.zeros((2, 3), dtype=np.uint8),
        positions=np.asarray([
            [[0, 0]], [[0, 1]], [[0, 0]], [[0, 1]], [[0, 1]],
        ], dtype=np.int16),
        goals=np.asarray([[0, 1]], dtype=np.int16),
        actions=np.zeros((4, 1), dtype=np.uint8),
    )
    assert episode.get_arrival_steps().tolist() == [3]
