import numpy as np

from mapf_transformer.geometry import crop_local_map
from mapf_transformer.spatial_memory import EgoSpatialMemory


def test_incremental_update_matches_full_crop():
    obstacles = np.zeros((20, 20), dtype=np.uint8)
    obstacles[3:7, 11] = 1
    memory = EgoSpatialMemory(15)
    first = memory.update(obstacles, (8, 8))
    assert first.full_refresh
    assert first.incoming_values.size == 225

    moved = memory.update(obstacles, (8, 9))
    assert not moved.reused
    assert not moved.full_refresh
    assert moved.incoming_values.size == 15
    np.testing.assert_array_equal(memory.snapshot(), crop_local_map(obstacles, (8, 9), 15))

    reused = memory.update(obstacles, (8, 9))
    assert reused.reused
    assert reused.incoming_values.size == 0
