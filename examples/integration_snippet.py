from __future__ import annotations

import torch

from mapf_map_transformer import AgentMapCrossAttention, ModelConfig, StructuredMapTransformer


batch_size, num_agents = 2, 25
halo_maps = torch.randint(0, 2, (batch_size, 17, 17), dtype=torch.long)
agent_tokens = torch.randn(batch_size, num_agents, 256)
agent_xy = torch.randint(-7, 8, (batch_size, num_agents, 2))

map_encoder = StructuredMapTransformer(ModelConfig())
map_tokens = map_encoder(halo_maps).latent_tokens

fusion = AgentMapCrossAttention(d_model=256, num_heads=8)
map_aware_agent_tokens = fusion(agent_tokens, map_tokens, agent_xy)

print("map_tokens:", tuple(map_tokens.shape))
print("map_aware_agent_tokens:", tuple(map_aware_agent_tokens.shape))
