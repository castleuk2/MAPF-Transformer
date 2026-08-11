from dataclasses import replace

import torch

from mapf_sst.communication import MultiRoundCommunicationPolicy
from mapf_sst.data.synthetic import make_synthetic_communication_group
from mapf_sst.model import SemanticSpatiotemporalPolicy


def test_batching_alone_does_not_communicate_but_wrapper_does(cfg):
    batch, graph = make_synthetic_communication_group(cfg, views=4, seed=9)
    base = SemanticSpatiotemporalPolicy(cfg).eval()
    wrapper = MultiRoundCommunicationPolicy(base).eval()
    with torch.no_grad():
        independent_a = base(batch).ego_logits[0]
        communicated_a = wrapper(batch, graph, rounds=1).final.ego_logits[0]

    changed = replace(batch, current_goal_delta=batch.current_goal_delta.clone())
    # Modify sender view 1 only; receiver view 0's own context is unchanged.
    changed.current_goal_delta[1] += 15
    with torch.no_grad():
        independent_b = base(changed).ego_logits[0]
        communicated_b = wrapper(changed, graph, rounds=1).final.ego_logits[0]
    assert torch.allclose(independent_a, independent_b, atol=1e-6)
    assert (communicated_a - communicated_b).abs().max() > 1.0e-8


def test_multi_round_gradients_reach_sender(cfg):
    batch, graph = make_synthetic_communication_group(cfg, views=4, seed=10)
    base = SemanticSpatiotemporalPolicy(cfg)
    wrapper = MultiRoundCommunicationPolicy(base)
    output = wrapper(batch, graph, rounds=2)
    loss = output.final.ego_logits[0].sum()
    loss.backward()
    assert base.message_query.grad is not None
    assert torch.isfinite(base.message_query.grad).all()
