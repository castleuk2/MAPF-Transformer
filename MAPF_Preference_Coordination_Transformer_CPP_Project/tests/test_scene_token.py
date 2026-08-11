from dataclasses import replace

import torch

from mapf_pct.config import ModelConfig
from mapf_pct.context_encoder import SceneTokenEncoder
from mapf_pct.data import make_synthetic_sample
from mapf_pct.types import stack_policy_batches


def test_scene_statistics_change_scene_token():
    cfg = ModelConfig(d_model=64, n_heads=4, coordination_layers=1, dropout=0.0)
    batch = stack_policy_batches([make_synthetic_sample(cfg, 5)])
    encoder = SceneTokenEncoder(cfg).eval()
    first = encoder(batch)
    changed = replace(batch, scene_numeric=batch.scene_numeric + 0.5)
    second = encoder(changed)
    assert first.shape == (1, 1, 64)
    assert not torch.allclose(first, second)
