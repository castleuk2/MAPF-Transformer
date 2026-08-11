from dataclasses import replace

import torch

from mapf_sst.constants import Action
from mapf_sst.data.synthetic import make_synthetic_policy_batch
from mapf_sst.model import SemanticSpatiotemporalPolicy
from mapf_sst.tokenizers import HistoryTokenizer


def test_invalid_agent_and_history_values_are_masked(cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=1, seed=4, num_current_agents=3)
    model = SemanticSpatiotemporalPolicy(cfg).eval()
    with torch.no_grad():
        reference = model(batch).ego_logits
    changed = replace(
        batch,
        current_xy=batch.current_xy.clone(),
        current_goal_delta=batch.current_goal_delta.clone(),
        history_xy=batch.history_xy.clone(),
        history_goal_delta=batch.history_goal_delta.clone(),
    )
    changed.current_xy[:, 3:] = 999
    changed.current_goal_delta[:, 3:] = -999
    changed.history_xy[~changed.history_valid] = 777
    changed.history_goal_delta[~changed.history_valid] = -777
    with torch.no_grad():
        result = model(changed).ego_logits
    assert torch.allclose(reference, result, atol=1e-6)


def test_history_preserves_lag_and_action_outcome(cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=1, seed=5, num_current_agents=4)
    batch.history_valid[:] = False
    batch.history_valid[:, 0, :] = True
    batch.history_selected_action[:] = int(Action.PAD)
    batch.history_observed_move[:] = int(Action.PAD)
    batch.history_selected_action[:, 0, 0] = int(Action.LEFT)
    batch.history_observed_move[:, 0, 0] = int(Action.WAIT)
    tokenizer = HistoryTokenizer(cfg).eval()
    with torch.no_grad():
        first = tokenizer(batch)
    swapped = replace(
        batch,
        history_selected_action=batch.history_selected_action.clone(),
        history_observed_move=batch.history_observed_move.clone(),
    )
    swapped.history_selected_action[:, 0, 0] = int(Action.PAD)
    swapped.history_observed_move[:, 0, 0] = int(Action.PAD)
    swapped.history_selected_action[:, 0, 3] = int(Action.LEFT)
    swapped.history_observed_move[:, 0, 3] = int(Action.WAIT)
    with torch.no_grad():
        second = tokenizer(swapped)
    # Same action/outcome pair at a different lag has a different embedding.
    assert not torch.allclose(first[:, 0, 0, 3], second[:, 0, 3, 3])


def test_current_history_same_track_mapping_affects_bias(cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=1, seed=6, num_current_agents=7)
    model = SemanticSpatiotemporalPolicy(cfg)
    bias_a = model.structure_bias(batch, dtype=torch.float32)
    changed = replace(
        batch,
        history_track_current_slot=batch.history_track_current_slot.clone(),
    )
    changed.history_track_current_slot[:, 1] = 5
    bias_b = model.structure_bias(changed, dtype=torch.float32)
    assert not torch.allclose(bias_a, bias_b)
