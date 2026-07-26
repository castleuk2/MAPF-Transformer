import pytest
import torch
from torch import nn

from mapf_transformer.config import ModelConfig
from mapf_transformer.dataset import SequenceSampleBuilder
from mapf_transformer.model import MAPFTransformer
from mapf_transformer.synthetic import generate_synthetic_episode


def _small_config(**overrides) -> ModelConfig:
    values = {
        "d_model": 32,
        "n_heads": 4,
        "temporal_layers": 1,
        "spatial_latent_layers": 1,
        "map_latents": 4,
        "dropout": 0.0,
        "mlp_ratio": 2,
        "one_hop_ctg": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_simplified_agent_metadata_requires_one_hop_ctg():
    config = ModelConfig(one_hop_ctg=False)
    with pytest.raises(ValueError, match="requires one_hop_ctg=true"):
        config.validate()


def test_model_forward_and_backward():
    config = _small_config(aux_map_reconstruction=True)
    episode = generate_synthetic_episode(seed=3, num_agents=2, max_steps=4)
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=0)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}

    model = MAPFTransformer(config)
    output = model(batch)
    assert output.logits.shape == (1, 5)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.action_head.weight.grad is not None
    assert model.agent_tokenizer.learned_agent_query.grad is not None
    assert model.agent_tokenizer.blocks[0].cross_attention.in_proj_weight.grad is not None


def test_graph_only_uses_current_frame_ego_token_without_temporal_transformer():
    config = _small_config(
        graph_only=True,
        history_frames=1,
        temporal_layers=0,
    )
    episode = generate_synthetic_episode(seed=13, num_agents=3, max_steps=4)
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=2)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = MAPFTransformer(config)

    frames, _, token_valid = model.encode_frames(batch)
    assert frames.shape[1:3] == (1, 25)
    assert token_valid.shape[1:] == (1, 25)
    assert config.context_tokens == 25
    assert model.transition_tokenizer is None
    assert len(model.temporal_blocks) == 0
    assert model.graph_only_block is not None

    output = model(batch)
    assert output.logits.shape == (1, 5)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.graph_only_block.attention.in_proj_weight.grad is not None
    assert model.action_head.weight.grad is not None


def test_eval_can_report_action_reconstruction_and_total_losses():
    config = _small_config(
        aux_map_reconstruction=True,
        aux_map_loss_weight=0.05,
    )
    episode = generate_synthetic_episode(seed=5, num_agents=2, max_steps=4)
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=0)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}

    model = MAPFTransformer(config).eval()
    default_output = model(batch)
    assert default_output.map_reconstruction_loss is None

    output = model(batch, return_reconstruction=True)
    assert output.action_loss is not None
    assert output.map_reconstruction_loss is not None
    assert output.loss is not None
    expected = output.action_loss + config.aux_map_loss_weight * output.map_reconstruction_loss
    assert torch.allclose(output.loss, expected)


@pytest.mark.parametrize(
    ("history_frames", "expected_context"),
    [(5, 131), (10, 261)],
)
def test_agent25_temporal_context_has_no_pre_temporal_agent_set_attention(
    history_frames: int,
    expected_context: int,
):
    config = _small_config(history_frames=history_frames)
    episode = generate_synthetic_episode(seed=7, num_agents=2, max_steps=4)
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=0)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = MAPFTransformer(config)

    frames, _, token_valid = model.encode_frames(batch)
    assert config.agents_per_frame == 25
    assert config.tokens_per_frame == 26
    assert config.context_tokens == expected_context
    assert frames.shape[1:3] == (history_frames, 26)
    assert token_valid.shape[1:] == (history_frames, 26)
    assert not hasattr(model, "agent_set_encoder")
    assert model.agent_tokenizer.field_type.num_embeddings == 8
    assert model.agent_tokenizer.learned_agent_query.shape == (1, 1, config.d_model)
    assert model.agent_tokenizer.stable_slot_embedding.num_embeddings == 25
    assert len(model.agent_tokenizer.one_hop_ctg_embeddings) == 4
    output = model(batch)
    assert output.logits.shape == (1, config.num_actions)


def test_temporal_mask_is_bidirectional_within_frame_and_causal_across_frames():
    config = _small_config(history_frames=5)
    model = MAPFTransformer(config)
    frame_valid = torch.ones((1, 5), dtype=torch.bool)
    token_valid = torch.ones((1, 5, config.tokens_per_frame), dtype=torch.bool)
    mask, _ = model._build_temporal_attention_mask(frame_valid, token_valid)
    # The mask is repeated for every attention head; inspect the first head.
    mask = mask[0]
    p = config.tokens_per_frame
    act = config.history_frames * p

    assert not mask[0, 1]       # Same-frame attention is bidirectional.
    assert mask[0, p]           # An old frame cannot read a future frame.
    assert not mask[p, 0]       # A newer frame can read an older frame.
    assert not mask[act, 0]     # ACT reads all valid temporal context.
    assert mask[0, act]         # Ordinary tokens cannot read ACT.


def test_interaction25_history5_builds_exactly_256_tokens_and_receives_gradients():
    config = _small_config(history_frames=5, interaction_latents=25)
    episode = generate_synthetic_episode(seed=11, num_agents=3, max_steps=4)
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=0)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = MAPFTransformer(config)

    frames, _, token_valid = model.encode_frames(batch)
    assert config.agents_per_frame == 25
    assert config.tokens_per_frame == 51
    assert config.context_tokens == 256
    assert frames.shape[1:3] == (5, 51)
    assert token_valid.shape[1:] == (5, 51)
    assert model.interaction_encoder is not None
    assert model.interaction_encoder.queries.shape == (1, 25, config.d_model)

    output = model(batch)
    assert output.loss is not None
    output.loss.backward()
    assert model.interaction_encoder.queries.grad is not None
    assert model.interaction_encoder.cross_attention.in_proj_weight.grad is not None


def test_graph_hybrid_masks_only_distant_same_frame_agent_pairs():
    config = _small_config(
        history_frames=5,
        interaction_latents=25,
        same_frame_graph_attention=True,
        graph_radius=3,
        graph_temporal_layers=1,
    )
    model = MAPFTransformer(config)
    frame_valid = torch.ones((1, 5), dtype=torch.bool)
    token_valid = torch.ones((1, 5, config.tokens_per_frame), dtype=torch.bool)
    agent_x = torch.zeros((1, 5, config.agents_per_frame), dtype=torch.long)
    agent_y = torch.zeros_like(agent_x)
    agent_x[:, :, 1] = 3
    agent_x[:, :, 2] = 4

    graph_mask, _ = model._build_temporal_attention_mask(
        frame_valid,
        token_valid,
        agent_x=agent_x,
        agent_y=agent_y,
        apply_same_frame_graph=True,
    )
    graph_mask = graph_mask[0]  # First attention head.
    p = config.tokens_per_frame
    interaction_index = config.agents_per_frame
    transition_index = p - 1

    assert not graph_mask[0, 1]  # Manhattan distance 3 is connected.
    assert graph_mask[0, 2]      # Manhattan distance 4 is disconnected.
    assert not graph_mask[0, interaction_index]
    assert not graph_mask[0, transition_index]
    assert not graph_mask[p, 2]  # Cross-frame causal attention stays dense.


def test_graph_hybrid_routes_graph_mask_to_only_the_configured_early_layers():
    config = _small_config(
        history_frames=2,
        temporal_layers=3,
        same_frame_graph_attention=True,
        graph_radius=3,
        graph_temporal_layers=2,
    )
    model = MAPFTransformer(config)

    class CaptureBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask = None

        def forward(self, x, attention_mask, token_valid):
            self.mask = attention_mask.detach().clone()
            return x

    captures = nn.ModuleList([CaptureBlock() for _ in range(3)])
    model.temporal_blocks = captures
    p = config.tokens_per_frame
    frames = torch.zeros((1, 2, p, config.d_model))
    frame_valid = torch.ones((1, 2), dtype=torch.bool)
    token_valid = torch.ones((1, 2, p), dtype=torch.bool)
    agent_x = torch.zeros((1, 2, config.agents_per_frame), dtype=torch.long)
    agent_y = torch.zeros_like(agent_x)
    agent_x[:, :, 1] = 4

    model.forward_encoded_frames(
        frames,
        frame_valid,
        frame_token_valid=token_valid,
        agent_x=agent_x,
        agent_y=agent_y,
    )
    assert captures[0].mask[0, 0, 1]
    assert captures[1].mask[0, 0, 1]
    assert not captures[2].mask[0, 0, 1]
