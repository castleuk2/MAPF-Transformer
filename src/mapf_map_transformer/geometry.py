from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from .actions import Action, coerce_action


DIRECTION_NAMES: Final[tuple[str, ...]] = ("north", "south", "west", "east")
NORTH, SOUTH, WEST, EAST = range(4)


@dataclass(slots=True)
class PatchGeometry:
    features: torch.Tensor
    patch_states: torch.Tensor
    port_open: torch.Tensor
    port_known: torch.Tensor
    outer_edge_mask: torch.Tensor
    connectivity_codes: torch.Tensor


def _as_batched_tensor(halo_maps: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if halo_maps.ndim == 2:
        return halo_maps.unsqueeze(0), True
    if halo_maps.ndim != 3:
        raise ValueError(f"Expected [17,17] or [B,17,17], got {tuple(halo_maps.shape)}")
    return halo_maps, False


def validate_halo_tensor(
    halo_maps: torch.Tensor,
    *,
    halo_size: int = 17,
    num_cell_states: int = 2,
) -> None:
    batched, _ = _as_batched_tensor(halo_maps)
    if batched.shape[-2:] != (halo_size, halo_size):
        raise ValueError(f"Expected halo shape {halo_size}x{halo_size}, got {tuple(batched.shape[-2:])}")
    if not torch.is_floating_point(batched):
        min_value = int(batched.min().item())
        max_value = int(batched.max().item())
        if min_value < 0 or max_value >= num_cell_states:
            raise ValueError(
                f"Cell states must be in [0,{num_cell_states - 1}], got min={min_value}, max={max_value}."
            )


def extract_core(halo_maps: torch.Tensor, halo_width: int = 1) -> torch.Tensor:
    batched, squeezed = _as_batched_tensor(halo_maps)
    core = batched[:, halo_width:-halo_width, halo_width:-halo_width]
    return core.squeeze(0) if squeezed else core


def patchify_core(core_maps: torch.Tensor, patch_size: int = 3) -> torch.Tensor:
    if core_maps.ndim == 2:
        core_maps = core_maps.unsqueeze(0)
        squeezed = True
    elif core_maps.ndim == 3:
        squeezed = False
    else:
        raise ValueError(f"Expected [H,W] or [B,H,W], got {tuple(core_maps.shape)}")
    batch, height, width = core_maps.shape
    if height != width or height % patch_size != 0:
        raise ValueError("Core map must be square and divisible by patch_size.")
    grid = height // patch_size
    patches = (
        core_maps.reshape(batch, grid, patch_size, grid, patch_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch, grid * grid, patch_size, patch_size)
    )
    return patches.squeeze(0) if squeezed else patches


def unpatchify(patches: torch.Tensor, grid_size: int = 5, patch_size: int = 3) -> torch.Tensor:
    if patches.ndim == 3:
        patches = patches.unsqueeze(0)
        squeezed = True
    elif patches.ndim == 4:
        squeezed = False
    else:
        raise ValueError(f"Expected [N,P,P] or [B,N,P,P], got {tuple(patches.shape)}")
    batch, num_patches, ph, pw = patches.shape
    if num_patches != grid_size * grid_size or (ph, pw) != (patch_size, patch_size):
        raise ValueError("Patch tensor shape does not match grid_size and patch_size.")
    maps = (
        patches.reshape(batch, grid_size, grid_size, patch_size, patch_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch, grid_size * patch_size, grid_size * patch_size)
    )
    return maps.squeeze(0) if squeezed else maps


def outer_edge_mask(grid_size: int, *, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros(grid_size, grid_size, 4, dtype=torch.bool, device=device)
    mask[0, :, NORTH] = True
    mask[-1, :, SOUTH] = True
    mask[:, 0, WEST] = True
    mask[:, -1, EAST] = True
    return mask.reshape(grid_size * grid_size, 4)


def compute_patch_ports(
    halo_maps: torch.Tensor,
    *,
    core_size: int = 15,
    patch_size: int = 3,
    free_state: int = 0,
    unknown_state: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 3-bit movement openings across every patch edge.

    The same definition is used for internal patch boundaries and the outside
    boundary of the 15x15 core. The one-cell halo supplies the neighboring cell
    for outer patch edges.
    """

    halo, squeezed = _as_batched_tensor(halo_maps)
    expected_halo = core_size + 2
    if halo.shape[-2:] != (expected_halo, expected_halo):
        raise ValueError(f"Expected {expected_halo}x{expected_halo} halo map.")
    batch = halo.shape[0]
    grid = core_size // patch_size
    free = halo.eq(free_state)
    known = torch.ones_like(free, dtype=torch.bool)
    if unknown_state is not None:
        known = halo.ne(unknown_state)

    port_open = torch.zeros(batch, grid * grid, 4, patch_size, dtype=torch.bool, device=halo.device)
    port_known = torch.zeros_like(port_open)

    token = 0
    for patch_row in range(grid):
        for patch_col in range(grid):
            row0 = 1 + patch_row * patch_size
            col0 = 1 + patch_col * patch_size
            row1 = row0 + patch_size - 1
            col1 = col0 + patch_size - 1

            src_n = free[:, row0, col0 : col0 + patch_size]
            nbr_n = free[:, row0 - 1, col0 : col0 + patch_size]
            kn_n = known[:, row0, col0 : col0 + patch_size] & known[:, row0 - 1, col0 : col0 + patch_size]

            src_s = free[:, row1, col0 : col0 + patch_size]
            nbr_s = free[:, row1 + 1, col0 : col0 + patch_size]
            kn_s = known[:, row1, col0 : col0 + patch_size] & known[:, row1 + 1, col0 : col0 + patch_size]

            src_w = free[:, row0 : row0 + patch_size, col0]
            nbr_w = free[:, row0 : row0 + patch_size, col0 - 1]
            kn_w = known[:, row0 : row0 + patch_size, col0] & known[:, row0 : row0 + patch_size, col0 - 1]

            src_e = free[:, row0 : row0 + patch_size, col1]
            nbr_e = free[:, row0 : row0 + patch_size, col1 + 1]
            kn_e = known[:, row0 : row0 + patch_size, col1] & known[:, row0 : row0 + patch_size, col1 + 1]

            port_open[:, token, NORTH] = src_n & nbr_n & kn_n
            port_open[:, token, SOUTH] = src_s & nbr_s & kn_s
            port_open[:, token, WEST] = src_w & nbr_w & kn_w
            port_open[:, token, EAST] = src_e & nbr_e & kn_e
            port_known[:, token, NORTH] = kn_n
            port_known[:, token, SOUTH] = kn_s
            port_known[:, token, WEST] = kn_w
            port_known[:, token, EAST] = kn_e
            token += 1

    if squeezed:
        return port_open.squeeze(0), port_known.squeeze(0)
    return port_open, port_known


def encode_three_bit_patterns(bits: torch.Tensor) -> torch.Tensor:
    if bits.shape[-1] != 3:
        raise ValueError("Three-bit pattern encoding requires a last dimension of size 3.")
    weights = torch.tensor([1, 2, 4], dtype=torch.long, device=bits.device)
    return (bits.to(torch.long) * weights).sum(dim=-1)


def build_patch_geometry(
    halo_maps: torch.Tensor,
    *,
    core_size: int = 15,
    patch_size: int = 3,
    num_cell_states: int = 2,
    free_state: int = 0,
    unknown_state: int | None = None,
    include_port_openings: bool = True,
    include_outer_edge_mask: bool = True,
    include_port_known_mask: bool = False,
) -> PatchGeometry:
    halo, squeezed = _as_batched_tensor(halo_maps)
    validate_halo_tensor(halo, halo_size=core_size + 2, num_cell_states=num_cell_states)
    core = extract_core(halo)
    patches = patchify_core(core, patch_size=patch_size)
    batch, num_patches, _, _ = patches.shape
    flat_states = patches.reshape(batch, num_patches, patch_size * patch_size).to(torch.long)
    state_one_hot = F.one_hot(flat_states, num_classes=num_cell_states).to(dtype=torch.float32)
    feature_parts = [state_one_hot.reshape(batch, num_patches, -1)]

    port_open, port_known = compute_patch_ports(
        halo,
        core_size=core_size,
        patch_size=patch_size,
        free_state=free_state,
        unknown_state=unknown_state,
    )
    grid = core_size // patch_size
    outer = outer_edge_mask(grid, device=halo.device).unsqueeze(0).expand(batch, -1, -1)

    if include_port_openings:
        feature_parts.append(port_open.to(torch.float32).reshape(batch, num_patches, -1))
    if include_outer_edge_mask:
        feature_parts.append(outer.to(torch.float32))
    if include_port_known_mask:
        feature_parts.append(port_known.to(torch.float32).reshape(batch, num_patches, -1))

    geometry = PatchGeometry(
        features=torch.cat(feature_parts, dim=-1),
        patch_states=flat_states,
        port_open=port_open,
        port_known=port_known,
        outer_edge_mask=outer,
        connectivity_codes=encode_three_bit_patterns(port_open),
    )
    if squeezed:
        return PatchGeometry(**{name: getattr(geometry, name).squeeze(0) for name in geometry.__dataclass_fields__})
    return geometry


def shift_halo_map(
    halo_map: np.ndarray,
    action: Action | int | str,
    incoming_strip: np.ndarray | list[int],
) -> np.ndarray:
    """Shift a 17x17 rolling halo map and insert exactly 17 new cells.

    The shift follows the *actual ego displacement*. If the ego moves UP, the
    new local crop gains a row at the top and all retained cells move down in
    local coordinates.
    """

    action = coerce_action(action)
    array = np.asarray(halo_map)
    if array.shape != (17, 17):
        raise ValueError(f"halo_map must have shape (17,17), got {array.shape}")
    strip = np.asarray(incoming_strip, dtype=array.dtype)
    if strip.shape != (17,):
        raise ValueError(f"incoming_strip must contain exactly 17 cells, got shape {strip.shape}")
    if action == Action.WAIT:
        raise ValueError("WAIT does not consume an incoming strip; reuse the existing map instead.")

    updated = np.empty_like(array)
    if action == Action.UP:
        updated[1:, :] = array[:-1, :]
        updated[0, :] = strip
    elif action == Action.DOWN:
        updated[:-1, :] = array[1:, :]
        updated[-1, :] = strip
    elif action == Action.LEFT:
        updated[:, 1:] = array[:, :-1]
        updated[:, 0] = strip
    elif action == Action.RIGHT:
        updated[:, :-1] = array[:, 1:]
        updated[:, -1] = strip
    else:  # pragma: no cover - coerce_action prevents this.
        raise AssertionError(action)
    return updated


def crop_global_map(
    global_map: np.ndarray,
    center_rc: tuple[int, int],
    *,
    size: int = 17,
    pad_value: int = 1,
) -> np.ndarray:
    grid = np.asarray(global_map)
    if grid.ndim != 2:
        raise ValueError("global_map must be a 2D array.")
    if size % 2 != 1:
        raise ValueError("size must be odd.")
    radius = size // 2
    row, col = center_rc
    result = np.full((size, size), pad_value, dtype=grid.dtype)

    src_r0 = max(0, row - radius)
    src_r1 = min(grid.shape[0], row + radius + 1)
    src_c0 = max(0, col - radius)
    src_c1 = min(grid.shape[1], col + radius + 1)

    dst_r0 = src_r0 - (row - radius)
    dst_c0 = src_c0 - (col - radius)
    result[dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = grid[
        src_r0:src_r1, src_c0:src_c1
    ]
    return result


def incoming_strip_from_global(
    global_map: np.ndarray,
    new_center_rc: tuple[int, int],
    action: Action | int | str,
    *,
    size: int = 17,
    pad_value: int = 1,
) -> np.ndarray:
    action = coerce_action(action)
    if action == Action.WAIT:
        raise ValueError("WAIT has no incoming strip.")
    full = crop_global_map(global_map, new_center_rc, size=size, pad_value=pad_value)
    if action == Action.UP:
        return full[0, :].copy()
    if action == Action.DOWN:
        return full[-1, :].copy()
    if action == Action.LEFT:
        return full[:, 0].copy()
    if action == Action.RIGHT:
        return full[:, -1].copy()
    raise AssertionError(action)
