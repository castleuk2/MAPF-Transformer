from __future__ import annotations

import torch

from mapf_map_transformer.config import ModelConfig
from mapf_map_transformer.fusion import AgentMapCrossAttention
from mapf_map_transformer.model import StructuredMapTransformer


def test_model_shapes_and_backward() -> None:
    config = ModelConfig(d_model=64, patch_hidden_dim=96, num_heads=4, dropout=0.0)
    model = StructuredMapTransformer(config)
    halo = torch.randint(0, 2, (3, 17, 17), dtype=torch.long)
    output = model(halo, return_attention=True)
    assert output.latent_tokens.shape == (3, 25, 64)
    assert output.reconstruction_logits.shape == (3, 15, 15)
    assert output.patch_logits.shape == (3, 25, 9)
    assert output.attention_maps is not None
    assert output.attention_maps[0].shape == (3, 4, 25, 25)
    output.reconstruction_logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_agent_map_fusion_shapes() -> None:
    fusion = AgentMapCrossAttention(d_model=64, num_heads=4, dropout=0.0)
    agents = torch.randn(2, 7, 64)
    maps = torch.randn(2, 25, 64)
    xy = torch.randint(-7, 8, (2, 7, 2))
    output = fusion(agents, maps, xy)
    assert output.shape == agents.shape
