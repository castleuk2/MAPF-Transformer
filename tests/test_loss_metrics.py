from __future__ import annotations

import torch

from mapf_map_transformer.config import LossConfig, ModelConfig
from mapf_map_transformer.losses import MapReconstructionLoss
from mapf_map_transformer.metrics import binary_reconstruction_metrics
from mapf_map_transformer.model import StructuredMapTransformer


def test_perfect_reconstruction_metrics() -> None:
    halo = torch.randint(0, 2, (2, 17, 17), dtype=torch.long)
    target = halo[:, 1:-1, 1:-1].to(torch.float32)
    logits = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    metrics = binary_reconstruction_metrics(logits, halo, include_reachability=True)
    assert metrics["cell_accuracy"] == 1.0
    assert metrics["occupied_iou"] == 1.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["reachability_iou"] == 1.0


def test_reconstruction_loss_is_finite() -> None:
    model_config = ModelConfig(d_model=32, patch_hidden_dim=48, num_heads=4, dropout=0.0)
    model = StructuredMapTransformer(model_config)
    halo = torch.randint(0, 2, (2, 17, 17), dtype=torch.long)
    output = model(halo)
    loss = MapReconstructionLoss(LossConfig(), model_config)(output, halo)
    assert torch.isfinite(loss.total)
    loss.total.backward()


def test_cell_ce_matches_direct_cross_entropy() -> None:
    model_config = ModelConfig(
        d_model=32, patch_hidden_dim=48, num_heads=4, dropout=0.0,
        reconstruction_classes=2,
    )
    model = StructuredMapTransformer(model_config)
    halo = torch.randint(0, 2, (2, 17, 17), dtype=torch.long)
    output = model(halo)
    loss = MapReconstructionLoss(LossConfig(mode="cell_ce"), model_config)(output, halo)
    expected = torch.nn.functional.cross_entropy(
        output.reconstruction_logits.reshape(-1, 2), halo[:, 1:-1, 1:-1].reshape(-1)
    )
    assert torch.allclose(loss.total, expected)
