import torch

from mapf_transformer.config import ModelConfig
from mapf_transformer.dataset import SequenceSampleBuilder
from mapf_transformer.model import MAPFTransformer
from mapf_transformer.synthetic import generate_synthetic_episode


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


def test_one_hop_ctg_is_optional_and_does_not_change_token_count():
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

    frames, _ = model.encode_frames(batch)
    assert frames.shape[1:3] == (config.history_frames, config.tokens_per_frame)
    output = model(batch)
    assert output.logits.shape == (1, config.num_actions)
    assert model.agent_tokenizer.one_hop_ctg_embeddings is not None
