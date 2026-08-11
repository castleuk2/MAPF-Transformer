from pathlib import Path

import torch

from mapf_map_transformer.config import experiment_config_from_dict
from mapf_map_transformer.model import StructuredMapTransformer
from mapf_sst.config import load_config
from mapf_sst.model import SemanticSpatiotemporalPolicy


PROJECT = Path(__file__).resolve().parents[1]


def test_native_pretrained_map_is_identical_and_frozen(monkeypatch):
    monkeypatch.chdir(PROJECT)
    cfg = load_config(PROJECT / "configs/base.yaml")
    policy = SemanticSpatiotemporalPolicy(cfg.model)

    checkpoint_path = Path(cfg.model.map_checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference = StructuredMapTransformer(
        experiment_config_from_dict(payload["config"]).model
    )
    reference.load_state_dict(payload["model_state"], strict=True)
    reference.eval()

    policy.train()
    assert policy.map_encoder.is_frozen
    assert not policy.map_encoder.encoder.training
    assert not any(parameter.requires_grad for parameter in policy.map_encoder.parameters())

    local_maps = torch.randint(0, 2, (2, 17, 17))
    with torch.no_grad():
        actual, _ = policy.map_encoder(local_maps)
        expected = reference(local_maps).latent_tokens
    assert torch.equal(actual, expected)
