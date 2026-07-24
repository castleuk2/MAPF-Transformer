from mapf_transformer.tracking import StableNeighborTracker


def test_full_tracker_replaces_farthest_agent_with_closer_newcomer():
    tracker = StableNeighborTracker(max_neighbors=2, grace_steps=1)
    first = tracker.update([1, 2], {1: 1.0, 2: 5.0})
    assert first.agent_ids.tolist() == [1, 2]

    updated = tracker.update([1, 2, 3], {1: 1.0, 2: 5.0, 3: 2.0})
    assert updated.agent_ids.tolist() == [1, 3]
    assert updated.valid.tolist() == [True, True]
    assert updated.reset.tolist() == [False, True]


def test_full_tracker_does_not_replace_with_farther_newcomer():
    tracker = StableNeighborTracker(max_neighbors=2, grace_steps=1)
    tracker.update([1, 2], {1: 1.0, 2: 2.0})

    updated = tracker.update([1, 2, 3], {1: 1.0, 2: 2.0, 3: 3.0})
    assert updated.agent_ids.tolist() == [1, 2]
    assert updated.reset.tolist() == [False, False]


def test_newcomer_reuses_missing_grace_slot_when_tracker_is_full():
    tracker = StableNeighborTracker(max_neighbors=2, grace_steps=1)
    tracker.update([1, 2], {1: 1.0, 2: 2.0})

    updated = tracker.update([1, 3], {1: 1.0, 3: 3.0})
    assert updated.agent_ids.tolist() == [1, 3]
    assert updated.reset.tolist() == [False, True]
