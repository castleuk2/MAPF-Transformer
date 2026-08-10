from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .data.npz_dataset import EpisodeFeatureBuilder, _Episode
from .resolver import CSPIBTResolver
from .types import stack_policy_batches


class PreferenceCoordinationPolicy:
    """All-egos preference inference followed by one global CS-PIBT resolver."""

    def __init__(self, checkpoint: str, device: str = "cuda:0", *, feature_backend: str = "python") -> None:
        if feature_backend not in {"python", "cpp"}:
            raise ValueError("feature_backend must be 'python' or 'cpp'")
        self.device = torch.device(device)
        self.model, _ = load_checkpoint(checkpoint, map_location="cpu")
        self.model = self.model.to(self.device).eval()
        self.builder = EpisodeFeatureBuilder(self.model.config)
        self.feature_backend = feature_backend
        if feature_backend == "cpp":
            from .cpp import CppEpisodeFeatureGenerator
            self.cpp_builder = CppEpisodeFeatureGenerator(self.model.config)
        else:
            self.cpp_builder = None
        self.resolver = CSPIBTResolver()
        self.last_timing: dict[str, float] = {}
        self._timing_total: defaultdict[str, float] = defaultdict(float)
        self._timed_steps = 0
        self.reset()

    def reset(self) -> None:
        self.position_history: list[np.ndarray] = []
        self.selected_history: list[np.ndarray] = []
        self.wait_age: np.ndarray | None = None
        self.last_timing = {}
        self._timing_total.clear()
        self._timed_steps = 0
        if self.cpp_builder is not None:
            self.cpp_builder.reset()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def timing_summary(self) -> dict[str, float]:
        """Return accumulated and per-step wall time for the profiled regions."""
        result: dict[str, float] = {"steps": float(self._timed_steps)}
        for name, total in self._timing_total.items():
            result[f"{name}_total_s"] = total
            result[f"{name}_mean_ms"] = 1000.0 * total / max(1, self._timed_steps)
        return result

    @torch.no_grad()
    def act_global(
        self,
        obstacles: np.ndarray,
        positions: np.ndarray,
        goals: np.ndarray,
    ) -> list[int]:
        step_started = perf_counter()
        positions = np.asarray(positions, dtype=np.int16)
        goals = np.asarray(goals, dtype=np.int16)
        if self.cpp_builder is not None:
            batch = self.cpp_builder.generate(obstacles, positions, goals)
        else:
            self.position_history.append(positions.copy())
            time_step = len(self.selected_history)
            action_rows = self.selected_history + [np.zeros(positions.shape[0], dtype=np.uint8)]
            episode = _Episode(
                obstacles=np.asarray(obstacles, dtype=np.uint8),
                positions=np.stack(self.position_history + [positions.copy()]),
                goals=goals,
                actions=np.stack(action_rows),
            )
            samples = [self.builder.build_episode(episode, time_step, ego) for ego in range(positions.shape[0])]
            batch = stack_policy_batches(samples)
        feature_done = perf_counter()

        batch = batch.to(self.device)
        self._synchronize()
        batch_done = perf_counter()

        output = self.model(batch, return_map_reconstruction=False)
        ego_logits = output.ego_logits
        selected = ego_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
        self._synchronize()
        forward_done = perf_counter()
        self.selected_history.append(selected)
        if self.cpp_builder is not None:
            self.cpp_builder.commit_actions(selected)

        on_goal = np.all(positions == goals, axis=1)
        if self.wait_age is None or self.wait_age.shape[0] != positions.shape[0]:
            self.wait_age = np.zeros(positions.shape[0], dtype=np.float32)
        preference = ego_logits.float().cpu()
        resolution = self.resolver.resolve(
            torch.from_numpy(positions.astype(np.int64)),
            preference,
            torch.from_numpy(np.asarray(obstacles, dtype=np.int64)),
            wait_age=torch.from_numpy(self.wait_age),
            on_goal=torch.from_numpy(on_goal),
        )
        actions = resolution.actions.numpy().astype(np.int64)
        self.wait_age = np.where(actions == 0, self.wait_age + 1.0, 0.0).astype(np.float32)
        resolver_done = perf_counter()

        self.last_timing = {
            "feature_s": feature_done - step_started,
            "batch_transfer_s": batch_done - feature_done,
            "forward_s": forward_done - batch_done,
            "resolver_s": resolver_done - forward_done,
            "total_s": resolver_done - step_started,
        }
        for name, elapsed in self.last_timing.items():
            self._timing_total[name] += elapsed
        self._timed_steps += 1
        return actions.tolist()
