import json

import numpy as np
import torch

from mapf_transformer.config import ModelConfig
from mapf_transformer.dataset import (
    EpisodeSequenceDataset,
    read_manifest,
    save_episode,
    write_manifest,
)
from mapf_transformer.model import MAPFTransformer
from mapf_transformer.packed_dataset import (
    PackedEpisodeSequenceDataset,
    _tracking_for_episode,
    build_packed_episode,
    packed_manifest_record,
    unpack_packed_batch,
    write_packed_manifest,
)
from mapf_transformer.synthetic import generate_synthetic_episode


def make_datasets(tmp_path, one_hop_ctg=False):
    config = ModelConfig(
        d_model=32,
        n_heads=4,
        temporal_layers=1,
        map_latents=4,
        one_hop_ctg=one_hop_ctg,
    )
    episode = generate_synthetic_episode(seed=7, num_agents=3, max_steps=5)
    ids, valid, reset = _tracking_for_episode(episode, config)
    episode.neighbor_ids = np.concatenate([ids, ids[-1:]], axis=0)
    episode.neighbor_valid = np.concatenate([valid, valid[-1:]], axis=0)
    episode.track_reset = np.concatenate([reset, reset[-1:]], axis=0)

    raw_path = tmp_path / "raw.npz"
    save_episode(raw_path, episode)
    raw_manifest = tmp_path / "raw_manifest.jsonl"
    write_manifest([raw_path], raw_manifest)
    source = read_manifest(raw_manifest)[0]

    packed_path = tmp_path / "packed.npz"
    np.savez(packed_path, **build_packed_episode(episode, config))
    packed_manifest = tmp_path / "packed_manifest.jsonl"
    record = packed_manifest_record(packed_path, packed_manifest, source, config)
    write_packed_manifest([record], packed_manifest)
    return (
        config,
        EpisodeSequenceDataset(raw_manifest, config, goal_wait_keep_ratio=0.2),
        PackedEpisodeSequenceDataset(packed_manifest, config, goal_wait_keep_ratio=0.2),
    )


def test_packed_cache_losslessly_restores_raw_features(tmp_path):
    config, raw, packed = make_datasets(tmp_path, one_hop_ctg=True)
    assert len(raw) == len(packed)
    keys = {
        "local_maps",
        "agent_x",
        "agent_y",
        "action_mask",
        "distance",
        "one_hop_ctg",
        "agent_valid",
        "track_reset",
        "previous_action",
        "actual_move",
        "outcome",
        "visible_count",
        "frame_valid",
        "target",
        "ego_id",
        "time_step",
    }
    for index in {0, len(raw) // 2, len(raw) - 1}:
        restored = unpack_packed_batch(packed[index], config)
        original = raw[index]
        for key in keys:
            torch.testing.assert_close(restored[key], original[key], rtol=0, atol=0)


def test_packed_and_raw_inputs_produce_identical_logits_and_losses(tmp_path):
    config, raw, packed = make_datasets(tmp_path)
    original = {key: value.unsqueeze(0) for key, value in raw[1].items()}
    restored = unpack_packed_batch(
        {key: value.unsqueeze(0) for key, value in packed[1].items()}, config
    )
    model = MAPFTransformer(config).eval()
    with torch.no_grad():
        raw_output = model(original, return_reconstruction=True)
        packed_output = model(restored, return_reconstruction=True)
    torch.testing.assert_close(raw_output.logits, packed_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(raw_output.loss, packed_output.loss, rtol=0, atol=0)


def test_packed_manifest_records_format_metadata(tmp_path):
    config, _, packed = make_datasets(tmp_path)
    record = json.loads((tmp_path / "packed_manifest.jsonl").read_text())
    assert record["packed_format"] == {
        "version": 1,
        "map_size": config.map_size,
        "max_neighbors": config.max_neighbors,
        "one_hop_ctg": False,
    }
    assert len(packed) > 0
