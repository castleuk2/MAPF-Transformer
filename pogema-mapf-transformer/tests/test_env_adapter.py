import numpy as np

from pogema_mapf_transformer.env_adapter import POGEMAMAPFTransformerAdapter


def test_evaluation_config_visualization_fields():
    from pogema_mapf_transformer.config import EvaluationConfig

    config = EvaluationConfig(render_mode="ansi", save_svg_dir="renders")
    config.validate()


class FakeEnv:
    def __init__(self):
        self.positions = np.asarray([[2, 2], [3, 3]], dtype=np.int16)
        self.goals = np.asarray([[4, 4], [1, 1]], dtype=np.int16)
        self.obstacles = np.zeros((6, 6), dtype=np.uint8)

    def observation(self):
        return [
            {
                "global_obstacles": self.obstacles,
                "global_xy": self.positions[agent_id].copy(),
                "global_target_xy": self.goals[agent_id].copy(),
            }
            for agent_id in range(len(self.positions))
        ]

    def reset(self):
        return self.observation(), {}

    def step(self, actions):
        # First commanded UP succeeds; second commanded RIGHT deliberately fails.
        if actions[0] == 1:
            self.positions[0, 0] -= 1
        obs = self.observation()
        return obs, [0.0, 0.0], [False, False], [False, False], [{}, {}]


def test_adapter_reports_actual_displacement_not_only_command():
    adapter = POGEMAMAPFTransformerAdapter(FakeEnv())
    adapter.reset()
    _, _, _, _, _, transition = adapter.step([1, 4])
    assert transition.actual_moves.tolist() == [1, 0]
    assert transition.outcomes.tolist() == [1, 2]
