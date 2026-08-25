from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mapf_pct.config import ModelConfig

cfg = ModelConfig()
print(f"Map       : {cfg.map_tokens}")
print(f"Agents    : {cfg.max_agents} × {cfg.agent_tokens_per_agent} = {cfg.agent_tokens}")
print(f"Relations : {cfg.relation_tokens}")
print(f"Events    : {cfg.event_tokens}")
print(f"Scene     : {cfg.scene_tokens}")
print(f"ACT       : {cfg.act_tokens}")
print(f"Total     : {cfg.total_tokens}")
