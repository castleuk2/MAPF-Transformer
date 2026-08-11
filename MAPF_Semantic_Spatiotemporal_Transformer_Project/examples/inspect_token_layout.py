from mapf_sst.config import ModelConfig
from mapf_sst.constants import TokenField
from mapf_sst.model import SemanticSpatiotemporalPolicy

cfg = ModelConfig(d_model=64, n_heads=4, transformer_layers=1)
model = SemanticSpatiotemporalPolicy(cfg)
print("total_tokens:", cfg.total_tokens)
print("map:", (0, cfg.current_offset - 1), cfg.map_tokens)
print("current:", (cfg.current_offset, cfg.history_offset - 1), cfg.current_tokens)
print("history:", (cfg.history_offset, cfg.coordination_offset - 1), cfg.history_tokens)
print("coordination:", (cfg.coordination_offset, cfg.total_tokens - 1), cfg.coordination_tokens)
for index in [0, 24, 25, 32, 136, 137, 248, 249, 255]:
    print(index, TokenField(int(model.field_ids[index])).name)
