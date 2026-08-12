import torch

from mapf_sst.constants import TokenField
from mapf_sst.config import ModelConfig
from mapf_sst.data.synthetic import make_synthetic_policy_batch
from mapf_sst.model import HierarchicalCandidatePolicy, SemanticSpatiotemporalPolicy


def test_fixed_256_layout(cfg):
    assert cfg.map_tokens == 25
    assert cfg.current_tokens == 112
    assert cfg.history_tokens == 112
    assert cfg.history_tracks == 14
    assert cfg.history_steps == 2
    assert cfg.coordination_tokens == 7
    assert cfg.total_tokens == 256
    model = SemanticSpatiotemporalPolicy(cfg)
    assert model.field_ids.numel() == 256
    assert model.position_embedding.num_embeddings == 256
    assert model.position_embedding.embedding_dim == cfg.d_model
    assert TokenField(int(model.field_ids[0])) is TokenField.MAP
    assert TokenField(int(model.field_ids[25])) is TokenField.CURRENT_POSITION
    assert TokenField(int(model.field_ids[32])) is TokenField.CANDIDATE_RIGHT
    assert TokenField(int(model.field_ids[137])) is TokenField.HISTORY_POSITION
    assert TokenField(int(model.field_ids[249])) is TokenField.MESSAGE_QUERY
    assert TokenField(int(model.field_ids[255])) is TokenField.NEIGHBOR_MESSAGE


def test_factorized_position_shares_current_history_track():
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        map_layers=1,
        transformer_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        token_position_mode="factorized_track",
        additive_embedding_init_std=0.02,
    )
    model = SemanticSpatiotemporalPolicy(cfg)
    assert model.position_embedding is None
    assert model.track_embedding is not None

    # Current agent 2 and History track 2 use the exact same learned row.
    current = cfg.current_offset + 2 * cfg.current_tokens_per_agent
    assert model.shared_track_ids[current : current + 8].unique().item() == 2
    for lag in range(cfg.history_steps):
        history = (
            cfg.history_offset
            + lag * cfg.history_tracks * cfg.history_tokens_per_step
            + 2 * cfg.history_tokens_per_step
        )
        assert model.shared_track_ids[history : history + 4].unique().item() == 2

    # Map/message positions use the zero PAD row; their own encoders provide
    # map 2-D position and message-slot identity.
    assert torch.count_nonzero(model.track_embedding.weight[-1]) == 0
    assert 0.01 < float(model.field_embedding.weight.std().detach()) < 0.03
    assert 0.01 < float(model.track_embedding.weight[:-1].std().detach()) < 0.03

    batch = make_synthetic_policy_batch(cfg, batch_size=2, seed=9)
    with torch.no_grad():
        output = model.eval()(batch, coordination_mode="none", return_tokens=True)
    assert output.final_tokens.shape == (2, 256, cfg.d_model)


def test_forward_shapes_and_candidate_readout(cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=2, seed=1, num_current_agents=5)
    model = SemanticSpatiotemporalPolicy(cfg).eval()
    with torch.no_grad():
        output = model(batch, coordination_mode="none", return_tokens=True, return_reconstruction=True)
    assert output.ego_logits.shape == (2, 5)
    assert output.all_current_logits.shape == (2, 14, 5)
    assert output.final_tokens.shape == (2, 256, cfg.d_model)
    assert output.token_padding_mask.shape == (2, 256)
    assert output.map_reconstruction_logits.shape == (2, 15, 15, 2)
    assert output.self_message is None
    # Ego output is the five fixed candidate positions in slot 0.
    assert torch.equal(output.ego_logits, output.all_current_logits[:, 0])


def test_query_and_message_slots(cfg):
    batch = make_synthetic_policy_batch(cfg, batch_size=2, seed=2)
    model = SemanticSpatiotemporalPolicy(cfg).eval()
    with torch.no_grad():
        none = model(batch, coordination_mode="none")
        query = model(batch, coordination_mode="query_only")
        messages = torch.randn(2, cfg.message_neighbors, cfg.d_model)
        valid = torch.ones(2, cfg.message_neighbors, dtype=torch.bool)
        source = torch.arange(1, cfg.message_neighbors + 1)[None].expand(2, -1)
        conditioned = model(
            batch,
            coordination_mode="messages",
            neighbor_messages=messages,
            neighbor_message_valid=valid,
            neighbor_message_source_slot=source,
        )
    assert none.token_padding_mask[:, 249:].all()
    assert not query.token_padding_mask[:, 249].any()
    assert query.token_padding_mask[:, 250:].all()
    assert query.self_message.shape == (2, cfg.d_model)
    assert not conditioned.token_padding_mask[:, 249:].any()


def test_hierarchical_candidate_forward_and_backward():
    cfg = ModelConfig(
        architecture="hierarchical_candidate",
        d_model=32,
        n_heads=4,
        map_layers=1,
        transformer_layers=1,
        candidate_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        enable_map_reconstruction=False,
        enable_semantic_reconstruction=False,
    )
    batch = make_synthetic_policy_batch(cfg, batch_size=2, seed=21, num_current_agents=10)
    model = HierarchicalCandidatePolicy(cfg)
    output = model(batch, coordination_mode="none", return_tokens=True)
    assert output.ego_logits.shape == (2, 5)
    assert output.all_current_logits.shape == (2, 14, 5)
    assert output.final_tokens.shape == (2, 70, cfg.d_model)
    assert output.semantic_reconstruction is None
    output.ego_logits.sum().backward()
    missing = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
