from pathlib import Path

import torch

from mapf_sst.checkpoint import load_checkpoint, save_checkpoint
from mapf_sst.config import ProjectConfig
from mapf_sst.data.synthetic import make_synthetic_policy_batch
from mapf_sst.losses import compute_loss
from mapf_sst.model import SemanticSpatiotemporalPolicy


def test_ego_only_loss_backward(cfg, train_cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=2, seed=11)
    model = SemanticSpatiotemporalPolicy(cfg)
    output = model(batch, return_reconstruction=True)
    loss = compute_loss(output, batch, cfg, train_cfg)
    loss.total.backward()
    assert loss.ego_action.item() > 0
    assert model.candidate_score.weight.grad is not None


def test_checkpoint_v3_roundtrip(cfg, train_cfg, tmp_path: Path):
    project = ProjectConfig(model=cfg, training=train_cfg)
    model = SemanticSpatiotemporalPolicy(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        config=project,
        epoch=1,
        step=2,
    )
    clone = SemanticSpatiotemporalPolicy(cfg)
    payload = load_checkpoint(path, model=clone)
    assert payload["format_version"] == 3
    for left, right in zip(model.parameters(), clone.parameters()):
        assert torch.equal(left, right)
