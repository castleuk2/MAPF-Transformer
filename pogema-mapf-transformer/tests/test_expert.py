from pathlib import Path

import numpy as np

from pogema_mapf_transformer.expert import MAPFLNS2Planner, PrioritizedTimeExpandedPlanner


def test_prioritized_planner_is_collision_free_and_reaches_goals():
    obstacles = np.zeros((7, 7), dtype=np.uint8)
    starts = np.asarray([[1, 1], [1, 5], [5, 3]], dtype=np.int16)
    goals = np.asarray([[5, 5], [5, 1], [1, 3]], dtype=np.int16)
    planner = PrioritizedTimeExpandedPlanner(horizon=40, retries=8, seed=4)
    plan = planner.plan(obstacles, starts, goals)
    positions = planner.validate_plan(starts, plan.actions, obstacles)
    assert np.all(positions[-1] == goals)
    assert plan.actions.shape[1] == 3


def test_mapf_lns2_path_parser_and_action_conversion(tmp_path: Path):
    output = tmp_path / "paths.txt"
    output.write_text(
        "Agent 0:(1,1)->(1,2)->(2,2)->\n"
        "Agent 1:(3,3)->(3,3)->\n",
        encoding="utf-8",
    )
    paths = MAPFLNS2Planner._read_paths(output, 2)
    actions = MAPFLNS2Planner._paths_to_actions(paths)
    assert actions.tolist() == [[4, 0], [2, 0]]


def test_mapf_lns2_writes_movingai_coordinates(tmp_path: Path):
    scenario = tmp_path / "instance.scen"
    MAPFLNS2Planner._write_scenario(
        scenario,
        "instance.map",
        (10, 12),
        np.asarray([[2, 3]], dtype=np.int16),
        np.asarray([[7, 8]], dtype=np.int16),
    )
    fields = scenario.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert fields[2:8] == ["12", "10", "3", "2", "8", "7"]


def test_mapf_lns2_bridge_invokes_binary_and_validates_plan(tmp_path: Path):
    binary = tmp_path / "fake_lns.py"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "args=sys.argv\n"
        "scenario=args[args.index('-a')+1]\n"
        "output=args[args.index('--outputPaths')+1]\n"
        "fields=open(scenario).read().splitlines()[1].split('\\t')\n"
        "sx,sy,gx,gy=map(int,fields[4:8])\n"
        "path=[(sy,sx)]\n"
        "while sy!=gy:\n"
        " sy += 1 if gy>sy else -1; path.append((sy,sx))\n"
        "while sx!=gx:\n"
        " sx += 1 if gx>sx else -1; path.append((sy,sx))\n"
        "open(output,'w').write('Agent 0:' + ''.join(f'({r},{c})->' for r,c in path) + '\\n')\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    planner = MAPFLNS2Planner(str(binary), cutoff_time=2, seed=9)
    obstacles = np.zeros((7, 7), dtype=np.uint8)
    starts = np.asarray([[1, 1]], dtype=np.int16)
    goals = np.asarray([[4, 5]], dtype=np.int16)
    result = planner.plan(obstacles, starts, goals)
    positions = PrioritizedTimeExpandedPlanner.validate_plan(starts, result.actions, obstacles)
    assert result.planner == "mapf_lns2"
    assert positions[-1, 0].tolist() == [4, 5]
