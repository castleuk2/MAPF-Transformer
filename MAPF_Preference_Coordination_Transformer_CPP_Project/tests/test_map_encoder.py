import torch

from mapf_pct.config import ModelConfig
from mapf_pct.map_encoder import StructuredPatchMapEncoder


def test_structured_map_raw_feature_and_reconstruction_shapes():
    cfg = ModelConfig(d_model=64, n_heads=4, coordination_layers=1, dropout=0.0)
    encoder = StructuredPatchMapEncoder(cfg)
    maps = torch.zeros(2, 17, 17, dtype=torch.long)
    maps[:, 0] = 1
    maps[:, -1] = 1
    raw, openings = encoder.extract_raw_features(maps)
    assert raw.shape == (2, 25, 34)
    assert openings.shape == (2, 5, 5, 4, 3)
    tokens, reconstruction = encoder(maps, return_reconstruction=True)
    assert tokens.shape == (2, 25, 64)
    assert reconstruction.shape == (2, 15, 15, 2)
