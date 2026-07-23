import torch
import pytest

from mapf_transformer.config import ModelConfig
from mapf_transformer.dataset import SequenceSampleBuilder
from mapf_transformer.model import MAPFTransformer
from mapf_transformer.synthetic import generate_synthetic_episode


def test_simplified_agent_metadata_requires_one_hop_ctg():
    config = ModelConfig(one_hop_ctg=False)
    with pytest.raises(ValueError, match="requires one_hop_ctg=true"):
        config.validate()


def test_model_forward_and_backward():
    config = ModelConfig(
        d_model=32,
        n_heads=4,
        temporal_layers=1,
        spatial_latent_layers=1,
        map_latents=4,
        dropout=0.0,
        mlp_ratio=2,
        aux_map_reconstruction=True,
    )
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


def test_eval_can_report_action_reconstruction_and_total_losses():
    config = ModelConfig(
        d_model=32,
        n_heads=4,
        temporal_layers=1,
        spatial_latent_layers=1,
        map_latents=4,
        dropout=0.0,
        mlp_ratio=2,
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


def test_agent_local_encoder_and_interaction_latents_make_265_token_context():
    config = ModelConfig(
        d_model=32,
        n_heads=4,
        temporal_layers=1,
        spatial_latent_layers=1,
        map_latents=4,
        dropout=0.0,
        mlp_ratio=2,
        one_hop_ctg=True,
    )
    episode = generate_synthetic_episode(seed=7, num_agents=2, max_steps=4)
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=0)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = MAPFTransformer(config)

    frames, _, token_valid = model.encode_frames(batch)
    assert frames.shape[1:3] == (config.history_frames, config.tokens_per_frame)
    assert token_valid.shape[1:] == (config.history_frames, config.tokens_per_frame)
    assert config.agents_per_frame == 25
    assert config.interaction_latents == 7
    assert config.agent_latents == 32
    assert config.context_tokens == 265
    output = model(batch)
    assert output.logits.shape == (1, config.num_actions)
    assert model.agent_tokenizer.field_type.num_embeddings == 8
    assert model.agent_tokenizer.learned_agent_query.shape == (1, 1, config.d_model)
    assert model.agent_tokenizer.stable_slot_embedding.num_embeddings == 25
    assert not hasattr(model.agent_tokenizer, "anchor_embedding")
    assert not hasattr(model.agent_tokenizer, "slot_embedding")
    assert hasattr(model.agent_tokenizer.blocks[0], "cross_attention")
    assert len(model.agent_tokenizer.one_hop_ctg_embeddings) == 4
    assert model.agent_set_encoder.interaction_queries.shape[0] == 7
    assert model.agent_tokenizer.one_hop_ctg_embeddings is not None
    assert not hasattr(model.agent_tokenizer, "direction_embedding")
    assert not hasattr(model.agent_tokenizer, "flexibility_embedding")
    assert not hasattr(model.agent_tokenizer, "role_embedding")
    assert not hasattr(model.agent_tokenizer, "validity_embedding")
    assert hasattr(model.agent_tokenizer, "track_reset_embedding")


def test_agent25_uses_only_real_agent_latents_and_makes_209_token_context():
    config = ModelConfig(
        d_model=32,
        n_heads=4,
        temporal_layers=1,
        spatial_latent_layers=1,
        map_latents=4,
        agent_latents=25,
        history_frames=8,
        dropout=0.0,
        mlp_ratio=2,
        one_hop_ctg=True,
    )
    episode = generate_synthetic_episode(seed=11, num_agents=2, max_steps=4)
    sample = SequenceSampleBuilder(config).build(episode, ego_id=0, time_step=0)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = MAPFTransformer(config)

    frames, _, token_valid = model.encode_frames(batch)
    assert config.agents_per_frame == 25
    assert config.interaction_latents == 0
    assert config.tokens_per_frame == 26
    assert config.context_tokens == 209
    assert frames.shape[1:3] == (8, 26)
    assert token_valid.shape[1:] == (8, 26)
    assert model.agent_set_encoder.interaction_queries.shape == (0, config.d_model)
    output = model(batch)
    assert output.logits.shape == (1, config.num_actions)
