import torch

from mapf_pct.config import ModelConfig, TrainingConfig
from mapf_pct.data import make_synthetic_sample
from mapf_pct.losses import compute_loss
from mapf_pct.model import PreferenceCoordinationTransformer
from mapf_pct.types import stack_policy_batches


def tiny_config() -> ModelConfig:
    return ModelConfig(
        d_model=64,
        n_heads=4,
        coordination_layers=2,
        map_layers=1,
        history_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )


def test_token_budget_and_forward_shapes():
    cfg = tiny_config()
    assert cfg.total_tokens == 256
    batch = stack_policy_batches([make_synthetic_sample(cfg, 1), make_synthetic_sample(cfg, 2)])
    model = PreferenceCoordinationTransformer(cfg)
    output = model(batch, return_tokens=True, return_map_reconstruction=True)
    assert output.final_tokens is not None
    assert output.final_tokens.shape == (2, 256, 64)
    assert output.all_agent_logits.shape == (2, 25, 5)
    assert output.ego_logits.shape == (2, 5)
    assert output.relation_index.shape == (2, 24, 2)
    assert output.map_reconstruction_logits.shape == (2, 15, 15, 2)


def test_backward_reaches_scene_encoder():
    cfg = tiny_config()
    batch = stack_policy_batches([make_synthetic_sample(cfg, 3)])
    model = PreferenceCoordinationTransformer(cfg)
    output = model(batch, return_map_reconstruction=True)
    loss = compute_loss(output, batch, cfg, TrainingConfig()).total
    loss.backward()
    grad = model.scene_encoder.numeric_mlp[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
