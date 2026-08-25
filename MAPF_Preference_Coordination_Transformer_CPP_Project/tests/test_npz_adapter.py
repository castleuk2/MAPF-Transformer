import numpy as np

from mapf_pct.config import ModelConfig
from mapf_pct.data import EpisodeFeatureBuilder


def test_npz_episode_adapter(tmp_path):
    obstacles = np.zeros((10, 10), dtype=np.uint8)
    positions = np.array(
        [
            [[4, 4], [4, 6]],
            [[4, 5], [4, 6]],
            [[4, 5], [5, 6]],
        ],
        dtype=np.int16,
    )
    goals = np.array([[4, 8], [8, 6]], dtype=np.int16)
    actions = np.array([[4, 0], [0, 2]], dtype=np.uint8)
    path = tmp_path / "episode.npz"
    np.savez(path, obstacles=obstacles, positions=positions, goals=goals, actions=actions)
    cfg = ModelConfig(d_model=64, n_heads=4, coordination_layers=1, dropout=0.0)
    sample = EpisodeFeatureBuilder(cfg).build(path, time_step=1, ego=0)
    assert sample.local_maps.shape == (17, 17)
    assert sample.agent_valid.sum() == 2
    assert sample.agent_xy[0].tolist() == [7, 7]
    assert sample.all_agent_actions[0].item() == 0
