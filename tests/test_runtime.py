from __future__ import annotations

import numpy as np

from mapf_map_transformer.actions import ACTION_TO_DELTA, Action
from mapf_map_transformer.config import ModelConfig
from mapf_map_transformer.model import StructuredMapTransformer
from mapf_map_transformer.runtime import GlobalMapWindowProvider, MapTokenRuntime, VectorizedMapTokenRuntime
from mapf_map_transformer.synthetic import generate_global_map


def _model() -> StructuredMapTransformer:
    return StructuredMapTransformer(ModelConfig(d_model=32, patch_hidden_dim=48, num_heads=4, dropout=0.0))


def test_wait_reuses_exact_latent_tensor() -> None:
    global_map = generate_global_map(2, size=41)
    provider = GlobalMapWindowProvider(global_map)
    center = (20, 20)
    runtime = MapTokenRuntime(_model(), device="cpu")
    first = runtime.reset(provider.crop(center))
    second = runtime.step(Action.WAIT, moved=False)
    assert second.reused
    assert runtime.encode_count == 1
    assert runtime.reuse_count == 1
    assert first.latent_tokens.data_ptr() == second.latent_tokens.data_ptr()


def test_runtime_incremental_update_matches_provider_crop() -> None:
    global_map = generate_global_map(5, size=45)
    provider = GlobalMapWindowProvider(global_map)
    center = (22, 22)
    runtime = MapTokenRuntime(_model(), device="cpu")
    runtime.reset(provider.crop(center))
    for action in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
        dr, dc = ACTION_TO_DELTA[action]
        center = (center[0] + dr, center[1] + dc)
        result = runtime.step(action, moved=True, incoming_strip=provider.incoming_strip(center, action))
        assert np.array_equal(result.halo_map, provider.crop(center))
        assert not result.reused
    assert runtime.encode_count == 5


def test_vectorized_runtime_encodes_only_changed_agents() -> None:
    global_map = generate_global_map(8, size=45)
    provider = GlobalMapWindowProvider(global_map)
    centers = [(20, 20), (22, 22), (24, 24)]
    runtime = VectorizedMapTokenRuntime(_model(), device="cpu")
    runtime.reset([provider.crop(center) for center in centers])
    new_center = (22, 23)
    outputs = runtime.step(
        [Action.WAIT, Action.RIGHT, Action.WAIT],
        [False, True, False],
        [None, provider.incoming_strip(new_center, Action.RIGHT), None],
    )
    assert [output.reused for output in outputs] == [True, False, True]
    assert runtime.encode_count == 4
    assert runtime.reuse_count == 2
