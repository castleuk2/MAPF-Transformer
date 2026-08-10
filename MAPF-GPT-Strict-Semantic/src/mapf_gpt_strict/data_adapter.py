import torch
import torch.nn.functional as F

from .continuous_tokenizer import EncodedObservation


ACTION_DELTAS = torch.tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)))


def encode_policy_batch(batch) -> EncodedObservation:
    """Vectorized conversion of the shared expert PolicyBatch to 256 continuous tokens."""
    device = batch.local_maps.device
    b = batch.local_maps.shape[0]
    feat = torch.zeros(b, 256, 16, device=device)
    agent = torch.full((b, 256), 13, dtype=torch.long, device=device)
    field = torch.full((b, 256), 10, dtype=torch.long, device=device)
    lag = torch.full((b, 256), 6, dtype=torch.long, device=device)
    role = torch.zeros((b, 256), dtype=torch.long, device=device)
    valid = torch.zeros((b, 256), dtype=torch.long, device=device)
    field[:, :25] = 0; valid[:, :25] = 1

    n = 13
    av = batch.agent_valid[:, :n]
    for slot in range(n):
        p = 25 + slot * 8
        feat[:, p, :2] = (batch.agent_xy[:, slot].float() - 7.0).clamp(-8, 8) / 8.0
        feat[:, p+1, :2] = batch.goal_delta[:, slot].float().clamp(-8, 8) / 8.0
        hops = batch.remaining_hops[:, slot].float()
        feat[:, p+2, 0] = hops.clamp(0, 32) / 32.0
        feat[:, p+2, 1] = (hops > 1023).float()
        for action in range(5):
            q = p + 3 + action
            state = torch.where(batch.candidate_static_free[:, slot, action], 0,
                     torch.ones_like(batch.candidate_delta_ctg[:, slot, action]))
            state = torch.where(batch.candidate_target_occupied[:, slot, action], 2, state)
            delta_id = batch.candidate_delta_ctg[:, slot, action]
            delta = torch.where(delta_id == 0, -torch.ones_like(delta_id),
                    torch.where(delta_id == 2, torch.ones_like(delta_id), torch.zeros_like(delta_id)))
            feat[:, q, 0] = (batch.candidate_static_free[:, slot, action] & batch.candidate_in_view[:, slot, action]).float()
            feat[:, q, 1] = delta.float()
            feat[:, q, 2:5] = F.one_hot(state.long().clamp(0, 2), 3).float()
        agent[:, p:p+8] = slot; field[:, p:p+8] = torch.arange(1, 9, device=device)
        lag[:, p:p+8] = 0; role[:, p:p+8] = 1 if slot == 0 else 2
        valid[:, p:p+8] = av[:, slot, None].long()

    # Current slot 0..5 and history track 0..5 have exact correspondence.
    executed = batch.history_executed[:, :6].clamp(0, 5)
    moves = ACTION_DELTAS.to(device)[executed]
    selected = batch.history_selected[:, :6].clamp(0, 5)
    hvalid = batch.history_valid[:, :6]
    for slot in range(6):
        current_pos = (batch.agent_xy[:, slot].float() - 7.0)
        current_goal = batch.goal_delta[:, slot].float()
        for step in range(5):
            p = 129 + (slot * 5 + step) * 4
            displacement = moves[:, slot, step:].sum(dim=1).float()
            feat[:, p, :2] = (current_pos - displacement).clamp(-8, 8) / 8.0
            feat[:, p+1, :2] = (current_goal + displacement).clamp(-8, 8) / 8.0
            future_delta = batch.history_delta_ctg[:, slot, step:]
            before_adjust = (future_delta == 0).sum(1) - (future_delta == 2).sum(1)
            hist_hops = batch.remaining_hops[:, slot].float() + before_adjust.float()
            feat[:, p+2, 0] = hist_hops.clamp(0, 32) / 32.0
            feat[:, p+2, 1] = (batch.remaining_hops[:, slot] > 1023).float()
            feat[:, p+3, :6] = F.one_hot(selected[:, slot, step], 6).float()
            feat[:, p+3, 6:12] = F.one_hot(executed[:, slot, step], 6).float()
            agent[:, p:p+4] = slot; field[:, p:p+4] = torch.tensor((1,2,3,9), device=device)
            lag[:, p:p+4] = 5-step; role[:, p:p+4] = 1 if slot == 0 else 2
            valid[:, p:p+4] = hvalid[:, slot, step, None].long()
    return EncodedObservation(feat, agent, field, lag, role, valid)

