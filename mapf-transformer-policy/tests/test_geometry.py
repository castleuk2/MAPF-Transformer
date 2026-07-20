import numpy as np

from mapf_transformer.geometry import (
    CTG_BLOCKED,
    CTG_DECREASE,
    CTG_INCREASE,
    CTG_SAME,
    bfs_distance_map,
    one_hop_cost_to_go,
    pack_agent_payload,
    quantize_distance,
    shortest_path_action_mask,
    unpack_agent_payload,
)


def test_multi_path_shortest_action_mask():
    obstacles = np.zeros((3, 3), dtype=np.uint8)
    distance = bfs_distance_map(obstacles, (0, 0))
    mask = shortest_path_action_mask(distance, (1, 1))
    # Order: UP, DOWN, LEFT, RIGHT. UP and LEFT both reduce distance.
    assert mask.tolist() == [1, 0, 1, 0]


def test_one_hop_cost_to_go_distinguishes_progress_detour_and_blocked():
    obstacles = np.zeros((3, 3), dtype=np.uint8)
    obstacles[1, 2] = 1
    distance = bfs_distance_map(obstacles, (0, 0))
    ctg = one_hop_cost_to_go(distance, obstacles, (1, 1))
    # WAIT, UP, DOWN, LEFT, RIGHT.
    assert ctg.tolist() == [
        CTG_SAME,
        CTG_DECREASE,
        CTG_INCREASE,
        CTG_DECREASE,
        CTG_BLOCKED,
    ]


def test_distance_quantization():
    assert quantize_distance(0) == 0
    assert quantize_distance(1) == 1
    assert quantize_distance(4) == 1
    assert quantize_distance(5) == 2
    assert quantize_distance(252) == 63
    assert quantize_distance(999) == 63


def test_payload_round_trip():
    payload = pack_agent_payload(7, 8, [1, 0, 1, 1], 42)
    x, y, mask, distance = unpack_agent_payload(payload)
    assert (x, y, distance) == (7, 8, 42)
    assert mask.tolist() == [1, 0, 1, 1]
