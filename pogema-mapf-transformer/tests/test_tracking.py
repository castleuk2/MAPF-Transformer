import numpy as np

from pogema_mapf_transformer.episode_io import precompute_neighbor_tracking


def test_tracking_is_stable_across_frames():
    # Three agents, two frames. Neighbor identities should keep their slots.
    positions = np.asarray(
        [
            [[5, 5], [5, 6], [6, 5]],
            [[5, 5], [5, 7], [7, 5]],
        ],
        dtype=np.int16,
    )
    ids, valid, reset = precompute_neighbor_tracking(positions, max_neighbors=2, radius=7)
    first_slots = ids[0, 0].copy()
    second_slots = ids[1, 0].copy()
    assert np.array_equal(first_slots, second_slots)
    assert valid[:, 0].all()
    assert reset[0, 0].all()
    assert not reset[1, 0].any()
