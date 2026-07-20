from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import crop_local_map


@dataclass(slots=True)
class SpatialUpdate:
    reused: bool
    full_refresh: bool
    displacement: tuple[int, int]
    incoming_local_indices: np.ndarray
    incoming_values: np.ndarray


class EgoSpatialMemory:
    """Persistent 15x15 Ego-centered static-map ring-buffer abstraction.

    It stores the complete current local map, but on a valid cardinal move only
    one incoming row/column (15 cells) is read from the global map.
    """

    def __init__(self, map_size: int = 15, outside_value: int = 1) -> None:
        if map_size <= 0 or map_size % 2 == 0:
            raise ValueError("map_size must be positive and odd")
        self.map_size = int(map_size)
        self.radius = map_size // 2
        self.outside_value = int(outside_value)
        self.buffer = np.full((map_size, map_size), outside_value, dtype=np.uint8)
        self.center_xy: np.ndarray | None = None
        self.initialized = False

    def reset(self) -> None:
        self.buffer.fill(self.outside_value)
        self.center_xy = None
        self.initialized = False

    def initialize(self, obstacles: np.ndarray, ego_xy: Iterable[int]) -> SpatialUpdate:
        self.buffer = crop_local_map(obstacles, ego_xy, self.map_size, self.outside_value)
        self.center_xy = np.asarray(tuple(ego_xy), dtype=np.int16)
        self.initialized = True
        indices = np.indices((self.map_size, self.map_size)).reshape(2, -1).T.astype(np.int16)
        return SpatialUpdate(
            reused=False,
            full_refresh=True,
            displacement=(0, 0),
            incoming_local_indices=indices,
            incoming_values=self.buffer.reshape(-1).copy(),
        )

    def _global_cell(self, obstacles: np.ndarray, global_x: int, global_y: int) -> int:
        if 0 <= global_x < obstacles.shape[0] and 0 <= global_y < obstacles.shape[1]:
            return int(obstacles[global_x, global_y])
        return self.outside_value

    def update(self, obstacles: np.ndarray, ego_xy: Iterable[int]) -> SpatialUpdate:
        obstacles = np.asarray(obstacles, dtype=np.uint8)
        new_center = np.asarray(tuple(ego_xy), dtype=np.int16)
        if not self.initialized or self.center_xy is None:
            return self.initialize(obstacles, new_center)

        delta = new_center.astype(np.int32) - self.center_xy.astype(np.int32)
        dx, dy = int(delta[0]), int(delta[1])
        if dx == 0 and dy == 0:
            return SpatialUpdate(
                reused=True,
                full_refresh=False,
                displacement=(0, 0),
                incoming_local_indices=np.empty((0, 2), dtype=np.int16),
                incoming_values=np.empty((0,), dtype=np.uint8),
            )

        if abs(dx) + abs(dy) != 1:
            return self.initialize(obstacles, new_center)

        incoming_indices: list[tuple[int, int]] = []
        incoming_values: list[int] = []
        n = self.map_size
        r = self.radius

        if dx == 1:  # Ego moved down: old rows shift up; add bottom row.
            self.buffer[:-1, :] = self.buffer[1:, :]
            local_row = n - 1
            global_row = int(new_center[0]) + r
            for col in range(n):
                value = self._global_cell(obstacles, global_row, int(new_center[1]) + col - r)
                self.buffer[local_row, col] = value
                incoming_indices.append((local_row, col))
                incoming_values.append(value)
        elif dx == -1:  # Ego moved up: old rows shift down; add top row.
            self.buffer[1:, :] = self.buffer[:-1, :]
            local_row = 0
            global_row = int(new_center[0]) - r
            for col in range(n):
                value = self._global_cell(obstacles, global_row, int(new_center[1]) + col - r)
                self.buffer[local_row, col] = value
                incoming_indices.append((local_row, col))
                incoming_values.append(value)
        elif dy == 1:  # Ego moved right: old columns shift left; add right column.
            self.buffer[:, :-1] = self.buffer[:, 1:]
            local_col = n - 1
            global_col = int(new_center[1]) + r
            for row in range(n):
                value = self._global_cell(obstacles, int(new_center[0]) + row - r, global_col)
                self.buffer[row, local_col] = value
                incoming_indices.append((row, local_col))
                incoming_values.append(value)
        else:  # dy == -1: Ego moved left; old columns shift right; add left column.
            self.buffer[:, 1:] = self.buffer[:, :-1]
            local_col = 0
            global_col = int(new_center[1]) - r
            for row in range(n):
                value = self._global_cell(obstacles, int(new_center[0]) + row - r, global_col)
                self.buffer[row, local_col] = value
                incoming_indices.append((row, local_col))
                incoming_values.append(value)

        self.center_xy = new_center
        return SpatialUpdate(
            reused=False,
            full_refresh=False,
            displacement=(dx, dy),
            incoming_local_indices=np.asarray(incoming_indices, dtype=np.int16),
            incoming_values=np.asarray(incoming_values, dtype=np.uint8),
        )

    def snapshot(self) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Spatial memory is not initialized")
        return self.buffer.copy()
