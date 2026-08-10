import torch

from mapf_gpt_strict import (
    Candidate, CSPIBTResolver, CurrentAgent, GPTConfig, HistoryState, MAPFGPT,
    SemanticTokenizer, StableHistoryBuffer, StableSlotAllocator,
)


def ego():
    candidates = tuple(Candidate(True, i - 2, 0) for i in range(5))
    return CurrentAgent((0, 0), (4, -3), 9, candidates)


def test_exact_layout_and_target():
    tok = SemanticTokenizer()
    history = [[HistoryState((0, 0), (4, -3), 9, 1, 1)]]
    slots = [ego()] + [None] * 12
    tracks = history + [[] for _ in range(5)]
    encoded = tok.encode_stable(slots, tracks)
    assert encoded.token_features.shape == (256, 16)
    assert torch.all(encoded.valid_ids[33:129] == 0)
    assert torch.all(encoded.valid_ids[129:145] == 0)
    assert torch.all(encoded.valid_ids[249:] == 0)


def test_model_preserves_mapf_gpt_io_contract():
    tok = SemanticTokenizer()
    encoded = tok.encode_stable([ego()] + [None] * 12, [[] for _ in range(6)]).unsqueeze(0)
    model = MAPFGPT(GPTConfig(n_layer=1, n_head=2, n_embd=32))
    logits, loss = model(encoded, torch.tensor([2]))
    assert logits.shape == (1, 5)
    assert loss.ndim == 0
    assert model.action_distribution(encoded).shape == (1, 5)


def test_exact_structured_map_path_is_integrated():
    tok = SemanticTokenizer()
    encoded = tok.encode_stable([ego()] + [None] * 12, [[] for _ in range(6)]).unsqueeze(0)
    model = MAPFGPT(GPTConfig(n_layer=1, n_head=2, n_embd=32))
    halo = torch.zeros(1, 17, 17, dtype=torch.long)
    output = model.map_encoder(halo)
    assert output.latent_tokens.shape == (1, 25, 32)
    assert output.patch_geometry.features.shape == (1, 25, 34)
    logits, loss = model(encoded, torch.tensor([2]), halo_maps=halo)
    assert logits.shape == (1, 5)
    assert loss.ndim == 0


def test_stable_agent_correspondence_and_semantic_embeddings():
    tok = SemanticTokenizer()
    allocator = StableSlotAllocator(missed_ttl=1)
    a0 = CurrentAgent((0, 0), (4, 0), 4, ego().candidates, agent_id=10)
    a1 = CurrentAgent((1, 0), (3, 0), 3, ego().candidates, agent_id=11)
    first = allocator.assign(10, [a0, a1])
    slot = next(i for i, agent in enumerate(first) if agent and agent.agent_id == 11)
    allocator.assign(10, [a0])
    third = allocator.assign(10, [a0, a1])
    assert third[slot].agent_id == 11

    buffer = StableHistoryBuffer()
    buffer.append(10, HistoryState((0, 0), (4, 0), 4, 1, 1))
    history = buffer.tracks_for(allocator.slot_to_id)
    encoded = tok.encode_stable(third, history).unsqueeze(0)
    assert encoded.token_features.shape == (1, 256, 16)
    assert encoded.agent_slots[0, 25].item() == 0
    assert encoded.role_ids[0, 25].item() == 1
    assert encoded.lag_ids[0, 148].item() == 1

    model = MAPFGPT(GPTConfig(n_layer=1, n_head=2, n_embd=32))
    halo = torch.zeros(1, 17, 17, dtype=torch.long)
    logits, loss = model(encoded, torch.tensor([1]), halo_maps=halo)
    assert logits.shape == (1, 5)
    assert loss.ndim == 0


def test_frozen_map_encoder_stays_in_eval_mode():
    model = MAPFGPT(GPTConfig(n_layer=1, n_head=2, n_embd=32, dropout=0.1))
    model.freeze_map_encoder()
    model.train()
    assert not model.map_encoder.training
    assert not any(parameter.requires_grad for parameter in model.map_encoder.parameters())


def test_stable_id_slots_survive_distance_reordering_and_short_absence():
    allocator = StableSlotAllocator(missed_ttl=1)
    first = allocator.assign_ids(0, [0, 1, 2], {0: 0, 1: 1, 2: 2})
    slot_one = first.index(1); slot_two = first.index(2)
    second = allocator.assign_ids(0, [0, 1, 2], {0: 0, 1: 8, 2: 1})
    assert second[slot_one] == 1 and second[slot_two] == 2
    missing = allocator.assign_ids(0, [0, 2], {0: 0, 2: 1})
    assert missing[slot_one] == 1
    returned = allocator.assign_ids(0, [0, 1, 2], {0: 0, 1: 2, 2: 1})
    assert returned[slot_one] == 1


def test_resolver_prevents_vertex_conflict():
    resolver = CSPIBTResolver()
    positions = torch.tensor([[1, 0], [1, 2]])
    obstacles = torch.zeros(3, 3)
    # Agent 0 prefers RIGHT and agent 1 prefers LEFT: both target (1, 1).
    logits = torch.tensor([[0.0, -1.0, -1.0, -1.0, 5.0], [0.0, -1.0, -1.0, 5.0, -1.0]])
    actions = resolver.resolve(positions, logits, obstacles)
    targets = positions.numpy() + torch.tensor(((0,0),(-1,0),(1,0),(0,-1),(0,1))).numpy()[actions]
    assert len({tuple(target) for target in targets}) == 2
