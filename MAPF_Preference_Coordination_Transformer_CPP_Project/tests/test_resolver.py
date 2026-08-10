import torch

from mapf_pct.resolver import CSPIBTResolver


def test_resolver_removes_same_target_conflict():
    resolver = CSPIBTResolver()
    positions = torch.tensor([[2, 1], [2, 3]])
    obstacles = torch.zeros(5, 5, dtype=torch.long)
    logits = torch.full((2, 5), -5.0)
    logits[0, 4] = 5.0  # RIGHT -> center
    logits[1, 3] = 5.0  # LEFT  -> center
    logits[:, 0] = 1.0  # WAIT fallback
    result = resolver.resolve(positions, logits, obstacles)
    assert len({tuple(x.tolist()) for x in result.targets}) == 2
    assert result.changed_from_argmax.sum() >= 1


def test_resolver_prevents_edge_swap():
    resolver = CSPIBTResolver()
    positions = torch.tensor([[2, 1], [2, 2]])
    obstacles = torch.zeros(5, 5, dtype=torch.long)
    logits = torch.full((2, 5), -5.0)
    logits[0, 4] = 5.0
    logits[1, 3] = 5.0
    logits[:, 0] = 2.0
    result = resolver.resolve(positions, logits, obstacles)
    resolver.assert_collision_free(positions, result.actions, obstacles)
    assert not (result.actions.tolist() == [4, 3])
