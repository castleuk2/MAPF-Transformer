import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_npz_to_arrow import previous_action_tokens, retained_goal_wait_steps


def test_retained_goal_wait_steps_match_policy_dataset_rule():
    assert retained_goal_wait_steps(8, 10, 0.0).tolist() == []
    assert retained_goal_wait_steps(8, 10, 0.2).tolist() == [8]
    assert retained_goal_wait_steps(5, 10, 1.0).tolist() == [5, 6, 7, 8, 9]


def test_previous_action_tokens_are_left_padded_and_bounded():
    path = np.asarray([[5, 5], [5, 6], [5, 6], [4, 6]], dtype=np.int16)
    assert previous_action_tokens(path, 0, 5) == ["n"] * 5
    assert previous_action_tokens(path, 3, 5) == ["n", "n", "r", "w", "u"]
