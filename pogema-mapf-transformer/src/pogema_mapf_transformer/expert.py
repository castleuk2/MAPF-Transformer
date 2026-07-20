from __future__ import annotations

import heapq
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from mapf_transformer.geometry import MOVES, WAIT, action_from_displacement


@dataclass(slots=True)
class PlanResult:
    actions: np.ndarray
    priority_order: list[int]
    planner: str


class ExpertPlanner(Protocol):
    def plan(self, obstacles: np.ndarray, starts: np.ndarray, goals: np.ndarray) -> PlanResult:
        ...


class PlanningFailure(RuntimeError):
    pass


class PrioritizedTimeExpandedPlanner:
    """Collision-free prioritized planner used as an included dataset fallback.

    It is deliberately simple and retained for tests and plan validation; the
    production dataset pipeline uses MAPF-LNS2.
    """

    def __init__(self, horizon: int = 192, retries: int = 16, seed: int = 0) -> None:
        self.horizon = int(horizon)
        self.retries = int(retries)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _plan_one(
        self,
        obstacles: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        vertex_reservations: dict[int, set[tuple[int, int]]],
        edge_reservations: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]],
    ) -> list[tuple[int, int]] | None:
        # Heap entries: f, g, x, y. State identity includes time g.
        heap: list[tuple[int, int, int, int]] = []
        heapq.heappush(heap, (self._manhattan(start, goal), 0, start[0], start[1]))
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {(start[0], start[1], 0): None}
        best_g: dict[tuple[int, int, int], int] = {(start[0], start[1], 0): 0}

        while heap:
            _, time_step, x, y = heapq.heappop(heap)
            state = (x, y, time_step)
            if best_g.get(state) != time_step:
                continue
            if (x, y) == goal:
                # Stay-at-target must be free for the remaining reservation horizon.
                if all(goal not in vertex_reservations.get(t, set()) for t in range(time_step, self.horizon + 1)):
                    path: list[tuple[int, int]] = []
                    cursor: tuple[int, int, int] | None = state
                    while cursor is not None:
                        path.append((cursor[0], cursor[1]))
                        cursor = parent[cursor]
                    path.reverse()
                    path.extend([goal] * (self.horizon + 1 - len(path)))
                    return path
            if time_step >= self.horizon:
                continue

            for action in range(5):
                dx, dy = MOVES[action]
                nx, ny = x + int(dx), y + int(dy)
                next_time = time_step + 1
                if not (0 <= nx < obstacles.shape[0] and 0 <= ny < obstacles.shape[1]):
                    continue
                if obstacles[nx, ny] != 0:
                    continue
                next_pos = (nx, ny)
                current_pos = (x, y)
                if next_pos in vertex_reservations.get(next_time, set()):
                    continue
                # Reject head-on edge swaps with an already planned agent.
                if (next_pos, current_pos) in edge_reservations.get(time_step, set()):
                    continue
                next_state = (nx, ny, next_time)
                if next_state in best_g:
                    continue
                best_g[next_state] = next_time
                parent[next_state] = state
                score = next_time + self._manhattan(next_pos, goal)
                # Tiny wait penalty resolves tie ordering without forbidding waits.
                if action == WAIT:
                    score += 1
                heapq.heappush(heap, (score, next_time, nx, ny))
        return None

    @staticmethod
    def _reserve(
        path: Sequence[tuple[int, int]],
        vertex_reservations: dict[int, set[tuple[int, int]]],
        edge_reservations: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]],
    ) -> None:
        for time_step, position in enumerate(path):
            vertex_reservations.setdefault(time_step, set()).add(position)
            if time_step + 1 < len(path):
                edge_reservations.setdefault(time_step, set()).add((position, path[time_step + 1]))

    @staticmethod
    def _paths_to_actions(paths: list[list[tuple[int, int]]]) -> np.ndarray:
        horizon = min(len(path) for path in paths) - 1
        num_agents = len(paths)
        actions = np.zeros((horizon, num_agents), dtype=np.uint8)
        for time_step in range(horizon):
            for agent_id, path in enumerate(paths):
                delta = np.asarray(path[time_step + 1]) - np.asarray(path[time_step])
                matches = np.flatnonzero(np.all(MOVES == delta, axis=1))
                if matches.size != 1:
                    raise PlanningFailure(f"Non-cardinal path transition: {delta.tolist()}")
                actions[time_step, agent_id] = int(matches[0])
        # Remove all-wait suffix while retaining at least one action frame.
        non_wait_rows = np.flatnonzero(np.any(actions != WAIT, axis=1))
        if non_wait_rows.size:
            actions = actions[: int(non_wait_rows[-1]) + 1]
        else:
            actions = actions[:1]
        return actions

    @staticmethod
    def validate_plan(
        starts: np.ndarray,
        actions: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(starts, dtype=np.int16).copy()
        history = [positions.copy()]
        for row in actions:
            next_positions = positions + MOVES[np.asarray(row, dtype=np.int64)]
            for position in next_positions:
                x, y = (int(v) for v in position)
                if not (0 <= x < obstacles.shape[0] and 0 <= y < obstacles.shape[1]):
                    raise PlanningFailure("Plan leaves map bounds")
                if obstacles[x, y] != 0:
                    raise PlanningFailure("Plan enters an obstacle")
            if len({tuple(position) for position in next_positions.tolist()}) != len(next_positions):
                raise PlanningFailure("Plan contains a vertex conflict")
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if np.array_equal(positions[i], next_positions[j]) and np.array_equal(
                        positions[j], next_positions[i]
                    ):
                        raise PlanningFailure("Plan contains an edge conflict")
            positions = next_positions.astype(np.int16)
            history.append(positions.copy())
        return np.asarray(history, dtype=np.int16)

    def plan(self, obstacles: np.ndarray, starts: np.ndarray, goals: np.ndarray) -> PlanResult:
        obstacles = np.asarray(obstacles, dtype=np.uint8)
        starts = np.asarray(starts, dtype=np.int16)
        goals = np.asarray(goals, dtype=np.int16)
        num_agents = starts.shape[0]
        base_order = list(range(num_agents))

        for attempt in range(max(1, self.retries)):
            if attempt == 0:
                order = base_order.copy()
            else:
                order = self.rng.permutation(num_agents).tolist()
            vertices: dict[int, set[tuple[int, int]]] = {}
            edges: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
            paths_by_agent: list[list[tuple[int, int]] | None] = [None] * num_agents
            success = True
            for agent_id in order:
                path = self._plan_one(
                    obstacles,
                    tuple(int(v) for v in starts[agent_id]),
                    tuple(int(v) for v in goals[agent_id]),
                    vertices,
                    edges,
                )
                if path is None:
                    success = False
                    break
                paths_by_agent[agent_id] = path
                self._reserve(path, vertices, edges)
            if success:
                paths = [path for path in paths_by_agent if path is not None]
                if len(paths) != num_agents:
                    continue
                actions = self._paths_to_actions(paths)  # type: ignore[arg-type]
                final_positions = self.validate_plan(starts, actions, obstacles)[-1]
                if np.all(final_positions == goals):
                    return PlanResult(actions=actions, priority_order=order, planner="prioritized")
        raise PlanningFailure(
            f"Prioritized planner failed after {self.retries} priority orders; "
            "increase horizon/retries or use MAPF-LNS2."
        )


class ExternalCommandPlanner:
    """Generic file-based bridge for an external expert.

    Command items may contain `{input}` and `{output}` placeholders. The input
    JSON contains obstacle/start/goal arrays. The output JSON must contain an
    `actions` array in POGEMA order.
    """

    def __init__(self, command: Sequence[str], timeout_seconds: float = 60.0) -> None:
        self.command = list(command)
        self.timeout_seconds = float(timeout_seconds)

    def plan(self, obstacles: np.ndarray, starts: np.ndarray, goals: np.ndarray) -> PlanResult:
        with tempfile.TemporaryDirectory(prefix="mapf_expert_") as directory:
            input_path = Path(directory) / "scenario.json"
            output_path = Path(directory) / "plan.json"
            input_path.write_text(
                json.dumps(
                    {
                        "obstacles": np.asarray(obstacles, dtype=np.uint8).tolist(),
                        "starts": np.asarray(starts, dtype=np.int16).tolist(),
                        "goals": np.asarray(goals, dtype=np.int16).tolist(),
                        "action_order": ["WAIT", "UP", "DOWN", "LEFT", "RIGHT"],
                    }
                ),
                encoding="utf-8",
            )
            command = [
                item.format(input=str(input_path), output=str(output_path)) for item in self.command
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                raise PlanningFailure(
                    f"External planner failed ({completed.returncode}): {completed.stderr.strip()}"
                )
            if not output_path.exists():
                raise PlanningFailure("External planner did not create the requested output file")
            result = json.loads(output_path.read_text(encoding="utf-8"))
            actions = np.asarray(result["actions"], dtype=np.uint8)
            if actions.ndim != 2 or actions.shape[1] != starts.shape[0]:
                raise PlanningFailure("External actions must have shape [T,N]")
            PrioritizedTimeExpandedPlanner.validate_plan(starts, actions, obstacles)
            return PlanResult(actions=actions, priority_order=list(range(starts.shape[0])), planner="external")


class MAPFLNS2Planner:
    """File bridge for the official MAPF-LNS2 command-line solver."""

    _coordinate = re.compile(r"\((-?\d+),(-?\d+)\)")

    def __init__(
        self,
        binary: str,
        cutoff_time: float = 10.0,
        init_algo: str = "PP",
        replan_algo: str = "PP",
        destroy_strategy: str = "Adaptive",
        neighbor_size: int = 8,
        max_iterations: int = 0,
        screen: int = 0,
        seed: int = 0,
    ) -> None:
        self.binary = str(binary)
        self.cutoff_time = float(cutoff_time)
        self.init_algo = str(init_algo)
        self.replan_algo = str(replan_algo)
        self.destroy_strategy = str(destroy_strategy)
        self.neighbor_size = int(neighbor_size)
        self.max_iterations = int(max_iterations)
        self.screen = int(screen)
        self.seed = int(seed)

    @staticmethod
    def _write_map(path: Path, obstacles: np.ndarray) -> None:
        obstacles = np.asarray(obstacles, dtype=np.uint8)
        rows = ["".join("@" if cell else "." for cell in row) for row in obstacles]
        path.write_text(
            f"type octile\nheight {obstacles.shape[0]}\nwidth {obstacles.shape[1]}\nmap\n"
            + "\n".join(rows)
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_scenario(
        path: Path,
        map_name: str,
        shape: tuple[int, int],
        starts: np.ndarray,
        goals: np.ndarray,
    ) -> None:
        height, width = shape
        lines = ["version 1"]
        for start, goal in zip(starts, goals):
            # MovingAI scenario coordinates are column (x), row (y).
            sy, sx = (int(v) for v in start)
            gy, gx = (int(v) for v in goal)
            lines.append(f"0\t{map_name}\t{width}\t{height}\t{sx}\t{sy}\t{gx}\t{gy}\t0")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def _read_paths(cls, path: Path, num_agents: int) -> list[list[tuple[int, int]]]:
        paths: list[list[tuple[int, int]] | None] = [None] * num_agents
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("Agent "):
                continue
            prefix, _, payload = line.partition(":")
            agent_id = int(prefix.split()[1])
            if not 0 <= agent_id < num_agents:
                raise PlanningFailure(f"MAPF-LNS2 returned invalid agent id {agent_id}")
            coordinates = [(int(row), int(col)) for row, col in cls._coordinate.findall(payload)]
            if not coordinates:
                raise PlanningFailure(f"MAPF-LNS2 returned an empty path for agent {agent_id}")
            paths[agent_id] = coordinates
        if any(agent_path is None for agent_path in paths):
            raise PlanningFailure("MAPF-LNS2 output is missing one or more agent paths")
        return [agent_path for agent_path in paths if agent_path is not None]

    @staticmethod
    def _paths_to_actions(paths: list[list[tuple[int, int]]]) -> np.ndarray:
        makespan = max(len(path) for path in paths) - 1
        actions = np.zeros((makespan, len(paths)), dtype=np.uint8)
        for agent_id, path in enumerate(paths):
            for step in range(makespan):
                current = path[min(step, len(path) - 1)]
                following = path[min(step + 1, len(path) - 1)]
                actions[step, agent_id] = action_from_displacement(
                    np.asarray(following, dtype=np.int16) - np.asarray(current, dtype=np.int16)
                )
        return actions

    def plan(self, obstacles: np.ndarray, starts: np.ndarray, goals: np.ndarray) -> PlanResult:
        binary = Path(self.binary).expanduser()
        if not binary.is_file():
            raise PlanningFailure(
                f"MAPF-LNS2 binary not found: {binary}. Build the official repository first."
            )
        with tempfile.TemporaryDirectory(prefix="mapf_lns2_") as directory:
            directory_path = Path(directory)
            map_path = directory_path / "instance.map"
            scenario_path = directory_path / "instance.scen"
            paths_path = directory_path / "paths.txt"
            self._write_map(map_path, obstacles)
            self._write_scenario(scenario_path, map_path.name, obstacles.shape, starts, goals)
            command = [
                str(binary), "-m", str(map_path), "-a", str(scenario_path),
                "-k", str(starts.shape[0]), "-t", str(self.cutoff_time),
                "--outputPaths", str(paths_path), "--solver", "LNS",
                "--initAlgo", self.init_algo, "--replanAlgo", self.replan_algo,
                "--destoryStrategy", self.destroy_strategy,
                "--neighborSize", str(self.neighbor_size),
                "--maxIterations", str(self.max_iterations),
                "--screen", str(self.screen), "--seed", str(self.seed),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.cutoff_time + 10.0,
            )
            if completed.returncode != 0 or not paths_path.exists():
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise PlanningFailure(f"MAPF-LNS2 failed ({completed.returncode}): {detail}")
            paths = self._read_paths(paths_path, starts.shape[0])
            for agent_id, path in enumerate(paths):
                if path[0] != tuple(int(v) for v in starts[agent_id]):
                    raise PlanningFailure(f"MAPF-LNS2 path {agent_id} has the wrong start")
                if path[-1] != tuple(int(v) for v in goals[agent_id]):
                    raise PlanningFailure(f"MAPF-LNS2 path {agent_id} does not reach its goal")
            actions = self._paths_to_actions(paths)
            PrioritizedTimeExpandedPlanner.validate_plan(starts, actions, obstacles)
            return PlanResult(
                actions=actions,
                priority_order=list(range(starts.shape[0])),
                planner="mapf_lns2",
            )
