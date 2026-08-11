from dataclasses import replace

import torch

from mapf_sst.data.synthetic import make_synthetic_policy_batch
from mapf_sst.map_encoder import StructuredPatchMapEncoder
from mapf_sst.model import SemanticSpatiotemporalPolicy


def test_structured_map_encoder(cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=2, seed=7)
    encoder = StructuredPatchMapEncoder(cfg)
    raw, openings = encoder.extract_raw_features(batch.local_maps)
    tokens, reconstruction = encoder(batch.local_maps, return_reconstruction=True)
    assert raw.shape == (2, 25, 34)
    assert openings.shape == (2, 5, 5, 4, 3)
    assert tokens.shape == (2, 25, cfg.d_model)
    assert reconstruction.shape == (2, 15, 15, 2)


def test_target_patch_changes_candidate_conditioning(cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=1, seed=8, num_current_agents=2)
    model = SemanticSpatiotemporalPolicy(cfg).eval()
    with torch.no_grad():
        map_tokens, _ = model.map_encoder(batch.local_maps)
        current, candidate = model.current_tokenizer(batch)
        first = model.map_fusion(
            candidate,
            map_tokens,
            batch.current_xy,
            batch.candidate_target_core_xy,
            batch.candidate_in_core,
            batch.current_valid,
        )
    changed = replace(
        batch,
        candidate_target_core_xy=batch.candidate_target_core_xy.clone(),
        candidate_in_core=batch.candidate_in_core.clone(),
    )
    changed.candidate_target_core_xy[:, 0, 1] = torch.tensor([0, 0])
    changed.candidate_in_core[:, 0, 1] = True
    with torch.no_grad():
        second = model.map_fusion(
            candidate,
            map_tokens,
            changed.current_xy,
            changed.candidate_target_core_xy,
            changed.candidate_in_core,
            changed.current_valid,
        )
    assert not torch.allclose(first[:, 0, 1], second[:, 0, 1])
