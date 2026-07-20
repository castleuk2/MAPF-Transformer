from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmark_compare import SUITES, load_scenarios, run_mapf_lns2_scenario


def test_official_mapf_gpt_scenario_counts():
    root = Path(__file__).parents[1] / "configs" / "mapf_gpt_eval"
    scenarios = load_scenarios(root, SUITES)
    counts = {
        suite: sum(scenario.suite == suite for scenario in scenarios)
        for suite in SUITES
    }
    assert counts == {
        "random": 768,
        "mazes": 768,
        "warehouse": 768,
        "movingai": 512,
        "puzzles": 480,
    }
    assert len(scenarios) == 3296


def test_mapf_lns2_smoke_on_official_scenario():
    root = Path(__file__).parents[1]
    scenario = load_scenarios(root / "configs" / "mapf_gpt_eval", ["random"])[0]
    metrics = run_mapf_lns2_scenario(
        scenario,
        binary=str(root / "external" / "MAPF-LNS2" / "lns"),
        cutoff_time=0.1,
        max_iterations=1,
    )
    assert metrics["status"] == "completed"
    assert metrics["CSR"] == 1.0
    assert metrics["ISR"] == 1.0
