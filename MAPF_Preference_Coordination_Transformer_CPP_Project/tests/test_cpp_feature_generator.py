import numpy as np
import torch

from mapf_pct.config import ModelConfig
from mapf_pct.cpp import CppEpisodeFeatureGenerator
from mapf_pct.data.npz_dataset import EpisodeFeatureBuilder, _Episode
from mapf_pct.types import stack_policy_batches


def test_cpp_generator_matches_python_runtime_features():
    obstacles = np.zeros((12, 13), dtype=np.uint8)
    obstacles[2:10, 6] = 1
    obstacles[6, 6] = 0
    positions = np.array([
        [[4, 3], [5, 3], [7, 8]],
        [[4, 4], [5, 4], [7, 8]],
        [[4, 5], [5, 5], [7, 9]],
    ], dtype=np.int64)
    goals = np.array([[4, 9], [5, 9], [7, 3]], dtype=np.int64)
    actions = np.array([[4, 4, 0], [4, 4, 4]], dtype=np.int64)
    cfg = ModelConfig(d_model=64, n_heads=4, coordination_layers=1, dropout=0.0)
    cpp = CppEpisodeFeatureGenerator(cfg)
    python = EpisodeFeatureBuilder(cfg)
    episode = _Episode(obstacles, positions, goals, actions)

    ignored = {"all_agent_actions", "action_soft_targets", "reason_labels", "reason_valid", "scene_risk_labels"}
    for step in range(actions.shape[0]):
        actual = cpp.generate(obstacles, positions[step], goals)
        expected = stack_policy_batches([python.build_episode(episode, step, ego) for ego in range(goals.shape[0])])
        for name, expected_tensor in expected.tensors():
            if name in ignored:
                continue
            actual_tensor = getattr(actual, name)
            if expected_tensor.is_floating_point():
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-6, atol=1e-6, msg=lambda m: f"{name}: {m}")
            else:
                assert torch.equal(actual_tensor, expected_tensor), name
        cpp.commit_actions(actions[step])
