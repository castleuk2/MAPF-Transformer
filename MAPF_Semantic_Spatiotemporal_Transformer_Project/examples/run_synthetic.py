import torch

from mapf_sst.config import ModelConfig
from mapf_sst.data.synthetic import make_synthetic_policy_batch
from mapf_sst.model import SemanticSpatiotemporalPolicy

cfg = ModelConfig(d_model=64, n_heads=4, transformer_layers=1, dropout=0.0)
batch = make_synthetic_policy_batch(cfg, batch_size=2, seed=3, num_current_agents=6)
model = SemanticSpatiotemporalPolicy(cfg).eval()
with torch.no_grad():
    output = model(batch, coordination_mode="none", return_tokens=True)
print("ego_logits:", tuple(output.ego_logits.shape))
print("all_current_logits:", tuple(output.all_current_logits.shape))
print("context:", tuple(output.final_tokens.shape))
print("padding per sample:", output.token_padding_mask.sum(dim=1).tolist())
