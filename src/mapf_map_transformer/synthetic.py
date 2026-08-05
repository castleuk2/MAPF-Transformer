from __future__ import annotations

import numpy as np


def _carve_line(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], width: int = 1) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        for offset in range(-(width // 2), width - width // 2):
            rr = np.clip(r0 + offset, 0, grid.shape[0] - 1)
            grid[rr, lo : hi + 1] = 0
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        for offset in range(-(width // 2), width - width // 2):
            cc = np.clip(c0 + offset, 0, grid.shape[1] - 1)
            grid[lo : hi + 1, cc] = 0
    else:
        raise ValueError("Only axis-aligned lines are supported.")


def _add_wall_with_gap(grid: np.ndarray, rng: np.random.Generator, horizontal: bool) -> None:
    size = grid.shape[0]
    if horizontal:
        row = int(rng.integers(1, size - 1))
        start = int(rng.integers(0, max(1, size // 3)))
        end = int(rng.integers(max(start + 3, 2 * size // 3), size))
        grid[row, start:end] = 1
        gap = int(rng.integers(start, end))
        grid[row, gap : min(gap + int(rng.integers(1, 3)), end)] = 0
    else:
        col = int(rng.integers(1, size - 1))
        start = int(rng.integers(0, max(1, size // 3)))
        end = int(rng.integers(max(start + 3, 2 * size // 3), size))
        grid[start:end, col] = 1
        gap = int(rng.integers(start, end))
        grid[gap : min(gap + int(rng.integers(1, 3)), end), col] = 0


def _add_room(grid: np.ndarray, rng: np.random.Generator) -> None:
    size = grid.shape[0]
    height = int(rng.integers(4, min(10, size - 1)))
    width = int(rng.integers(4, min(10, size - 1)))
    r0 = int(rng.integers(0, size - height))
    c0 = int(rng.integers(0, size - width))
    r1, c1 = r0 + height - 1, c0 + width - 1
    grid[r0, c0 : c1 + 1] = 1
    grid[r1, c0 : c1 + 1] = 1
    grid[r0 : r1 + 1, c0] = 1
    grid[r0 : r1 + 1, c1] = 1
    side = int(rng.integers(0, 4))
    if side == 0:
        grid[r0, int(rng.integers(c0 + 1, c1))] = 0
    elif side == 1:
        grid[r1, int(rng.integers(c0 + 1, c1))] = 0
    elif side == 2:
        grid[int(rng.integers(r0 + 1, r1)), c0] = 0
    else:
        grid[int(rng.integers(r0 + 1, r1)), c1] = 0


def generate_halo_map(
    rng: np.random.Generator,
    *,
    size: int = 17,
    pattern: str = "mixed",
    min_density: float = 0.08,
    max_density: float = 0.38,
    ensure_center_free: bool = True,
) -> np.ndarray:
    if size != 17:
        raise ValueError("This project currently defines a 17x17 halo input.")
    if pattern == "mixed":
        pattern = str(rng.choice(["bernoulli", "walls", "rooms", "corridors"]))

    density = float(rng.uniform(min_density, max_density))
    if pattern == "bernoulli":
        grid = (rng.random((size, size)) < density).astype(np.uint8)
        # Add a few correlated segments so reconstruction is not only iid noise.
        for _ in range(int(rng.integers(1, 4))):
            _add_wall_with_gap(grid, rng, bool(rng.integers(0, 2)))
    elif pattern == "walls":
        grid = (rng.random((size, size)) < density * 0.25).astype(np.uint8)
        for _ in range(int(rng.integers(3, 8))):
            _add_wall_with_gap(grid, rng, bool(rng.integers(0, 2)))
    elif pattern == "rooms":
        grid = (rng.random((size, size)) < density * 0.15).astype(np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            _add_room(grid, rng)
    elif pattern == "corridors":
        grid = np.ones((size, size), dtype=np.uint8)
        center = size // 2
        row, col = center, center
        grid[row, col] = 0
        for _ in range(int(rng.integers(8, 20))):
            if rng.random() < 0.5:
                new_col = int(np.clip(col + rng.integers(-7, 8), 0, size - 1))
                _carve_line(grid, (row, col), (row, new_col), width=int(rng.integers(1, 3)))
                col = new_col
            else:
                new_row = int(np.clip(row + rng.integers(-7, 8), 0, size - 1))
                _carve_line(grid, (row, col), (new_row, col), width=int(rng.integers(1, 3)))
                row = new_row
        # Carve a few small waiting areas.
        for _ in range(int(rng.integers(1, 4))):
            rr = int(rng.integers(1, size - 2))
            cc = int(rng.integers(1, size - 2))
            grid[rr - 1 : rr + 2, cc - 1 : cc + 2] = 0
    else:
        raise ValueError(f"Unsupported synthetic pattern: {pattern}")

    if ensure_center_free:
        center = size // 2
        grid[center, center] = 0
    return grid


def generate_global_map(
    seed: int,
    *,
    size: int = 49,
    density: float = 0.22,
) -> np.ndarray:
    """Generate a larger static map for rolling-buffer demonstrations."""
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size)) < density * 0.2).astype(np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    for _ in range(max(8, size // 3)):
        _add_wall_with_gap(grid, rng, bool(rng.integers(0, 2)))
    center = size // 2
    grid[center - 2 : center + 3, center - 2 : center + 3] = 0
    return grid
