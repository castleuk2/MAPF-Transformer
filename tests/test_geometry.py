from __future__ import annotations

import numpy as np
import torch

from mapf_map_transformer.actions import ACTION_TO_DELTA, Action
from mapf_map_transformer.geometry import (
    compute_patch_ports,
    crop_global_map,
    extract_core,
    incoming_strip_from_global,
    patchify_core,
    shift_halo_map,
    unpatchify,
)


def test_patchify_round_trip() -> None:
    core = torch.arange(15 * 15).reshape(15, 15)
    patches = patchify_core(core, patch_size=3)
    assert patches.shape == (25, 3, 3)
    restored = unpatchify(patches, grid_size=5, patch_size=3)
    assert torch.equal(core, restored)


def test_halo_controls_outer_ports() -> None:
    halo = torch.zeros(1, 17, 17, dtype=torch.long)
    halo[:, 0, :] = 1
    ports, known = compute_patch_ports(halo)
    ports = ports.reshape(1, 5, 5, 4, 3)
    assert not ports[:, 0, :, 0].any()  # north outer continuation is blocked by halo.
    assert ports[:, 1:, :, 0].all()  # internal north boundaries remain open.
    assert known.all()


def test_incremental_strip_update_matches_fresh_crop() -> None:
    rng = np.random.default_rng(12)
    global_map = (rng.random((45, 45)) < 0.25).astype(np.uint8)
    center = (22, 22)
    current = crop_global_map(global_map, center)
    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        dr, dc = ACTION_TO_DELTA[action]
        new_center = (center[0] + dr, center[1] + dc)
        strip = incoming_strip_from_global(global_map, new_center, action)
        updated = shift_halo_map(current, action, strip)
        expected = crop_global_map(global_map, new_center)
        assert np.array_equal(updated, expected)


def test_core_is_center_15_by_15() -> None:
    halo = torch.arange(17 * 17).reshape(17, 17)
    core = extract_core(halo)
    assert core.shape == (15, 15)
    assert core[0, 0] == halo[1, 1]
    assert core[-1, -1] == halo[-2, -2]
