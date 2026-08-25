from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mapf_pct.config import ModelConfig
from mapf_pct.data import make_synthetic_sample
from mapf_pct.model import PreferenceCoordinationTransformer
from mapf_pct.types import stack_policy_batches

cfg = ModelConfig(d_model=64, n_heads=4, coordination_layers=2, dropout=0.0)
model = PreferenceCoordinationTransformer(cfg).eval()
batch = stack_policy_batches([make_synthetic_sample(cfg, seed=1)])
output = model(batch, return_tokens=True)
print("all-agent logits:", tuple(output.all_agent_logits.shape))
print("ego logits:", tuple(output.ego_logits.shape))
print("policy context:", tuple(output.final_tokens.shape))
