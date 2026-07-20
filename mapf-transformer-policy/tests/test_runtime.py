import numpy as np

from mapf_transformer.config import ModelConfig
from mapf_transformer.model import MAPFTransformer
from mapf_transformer.runtime import MAPFTransformerInference, MAPFTransformerInferenceConfig


def make_observations(positions):
    obstacles = np.zeros((17, 17), dtype=np.uint8)
    goals = np.asarray([[12, 12], [3, 3]], dtype=np.int16)
    positions = np.asarray(positions, dtype=np.int16)
    return [
        {
            "global_obstacles": obstacles,
            "global_xy": positions[agent_id],
            "global_target_xy": goals[agent_id],
        }
        for agent_id in range(len(positions))
    ]


def test_mapf_gpt_like_runtime_interface():
    config = ModelConfig(
        d_model=32,
        n_heads=4,
        temporal_layers=1,
        spatial_latent_layers=1,
        map_latents=4,
        dropout=0.0,
        mlp_ratio=2,
        aux_map_reconstruction=False,
    )
    policy = MAPFTransformerInference(
        MAPFTransformerInferenceConfig(device="cpu", sample_actions=False),
        model=MAPFTransformer(config),
    )
    actions0 = policy.act(make_observations([[8, 8], [9, 8]]))
    assert len(actions0) == 2
    assert all(0 <= action <= 4 for action in actions0)

    actions1 = policy.act(make_observations([[8, 8], [9, 9]]))
    assert len(actions1) == 2
    policy.reset_states()
