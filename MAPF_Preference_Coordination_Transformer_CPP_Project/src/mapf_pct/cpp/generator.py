from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

from ..config import ModelConfig
from ..constants import Action, DeltaCTG
from ..types import PolicyBatch


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    # pip places the ninja executable next to the active Python interpreter.
    # Ensure it is discoverable even when that interpreter was invoked by an
    # absolute path without activating its environment first.
    python_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    path = Path(__file__).with_name("_feature_generator.cpp")
    _EXTENSION = load(
        name="mapf_pct_feature_generator",
        sources=[str(path)],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=False,
        verbose=False,
    )
    return _EXTENSION


class CppEpisodeFeatureGenerator:
    """Stateful all-Ego C++ feature generator for MPCT runtime inference."""

    def __init__(self, config: ModelConfig, *, task_mode: str = "one_shot", target_mode: str = "stay") -> None:
        self.config = config
        self.task_mode_id = 0 if task_mode == "one_shot" else 1
        self.target_mode_id = 0 if target_mode == "stay" else 1
        self._module = None
        self._native = None
        self._signature: tuple | None = None

    def reset(self) -> None:
        self._native = None
        self._signature = None

    def _ensure(self, obstacles: np.ndarray, goals: np.ndarray) -> None:
        obstacles = np.ascontiguousarray(obstacles, dtype=np.uint8)
        goals = np.ascontiguousarray(goals, dtype=np.int64)
        signature = (obstacles.shape, obstacles.tobytes(), goals.shape, goals.tobytes())
        if signature == self._signature:
            return
        if self._module is None:
            self._module = _load_extension()
        cfg = self.config
        self._native = self._module.StatefulFeatureGenerator(
            obstacles, goals, cfg.core_map_size, cfg.local_map_size, cfg.max_agents,
            cfg.history_steps, cfg.max_hops, cfg.goal_delta_clip, cfg.contender_buckets,
        )
        self._signature = signature

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        return torch.from_numpy(np.asarray(value))

    def generate_with_slot_ids(
        self, obstacles: np.ndarray, positions: np.ndarray, goals: np.ndarray
    ) -> tuple[PolicyBatch, torch.Tensor]:
        """Return all-Ego features and the local-slot to global-agent mapping."""
        self._ensure(obstacles, goals)
        assert self._native is not None
        raw = {name: self._tensor(value) for name, value in self._native.generate(
            np.ascontiguousarray(positions, dtype=np.int64)
        ).items()}
        cfg = self.config
        valid = raw["agent_valid"].bool()
        history_valid = raw["history_valid"].bool()
        history_selected = raw["history_selected"].long()
        history_executed = raw["history_executed"].long()
        history_outcome = raw["history_outcome"].long()
        history_delta = raw["history_delta_ctg"].long()
        candidate_contenders = raw["candidate_contenders"].long()
        candidate_edge_swap = raw["candidate_edge_swap"].bool()
        on_goal = raw["on_goal"].bool()
        b, _, h = history_selected.shape

        event_action = history_selected[:, 0].clone()
        event_executed = history_executed[:, 0].clone()
        event_outcome = history_outcome[:, 0].clone()
        event_valid = history_valid[:, 0].clone()
        event_numeric = torch.zeros(b, h, cfg.event_numeric_dim)
        valid_count = valid.sum(dim=1)
        for step in range(h):
            mask = history_valid[:, :, step] & valid
            denom = mask.sum(dim=1).clamp_min(1).float()
            event_numeric[:, step, 0] = (((history_executed[:, :, step] != int(Action.WAIT)) & mask).sum(dim=1) / denom)
            event_numeric[:, step, 1] = (((history_selected[:, :, step] == int(Action.WAIT)) & mask).sum(dim=1) / denom)
            event_numeric[:, step, 2] = (((history_selected[:, :, step] != history_executed[:, :, step]) & mask).sum(dim=1) / denom)
            event_numeric[:, step, 3] = valid_count.float() / cfg.max_agents
            event_numeric[:, step, 4] = (((history_delta[:, :, step] == int(DeltaCTG.DECREASE)) & mask).sum(dim=1) / denom)
            event_numeric[:, step, 5] = candidate_contenders.amax(dim=(1, 2)).float() / max(1, cfg.contender_buckets - 1)
            event_numeric[:, step, 6] = (candidate_edge_swap.float() * valid[:, :, None]).sum(dim=(1, 2)) / (valid_count.clamp_min(1) * cfg.num_actions)
            event_numeric[:, step, 7] = (on_goal & valid).sum(dim=1) / valid_count.clamp_min(1)
            event_numeric[:, step] *= mask.any(dim=1)[:, None]

        remaining = raw["remaining_hops"].long()
        csf = raw["candidate_static_free"].bool()
        cgr = raw["candidate_greedy"].bool()
        cbn = raw["candidate_bottleneck"].bool()
        ccg = raw["candidate_congestion"].float()
        # Match EpisodeFeatureBuilder's scalar reduction order exactly. The
        # vectorized form differs by one float32 ULP for some means.
        scene_rows = []
        for ego in range(b):
            count = int(valid_count[ego])
            finite_hops = remaining[ego, :count].float().clamp_max(cfg.max_hops) / max(1, cfg.max_hops)
            core = raw["local_maps"][ego, 1:-1, 1:-1]
            scene_rows.append(torch.tensor((
                float(core.float().mean()), count / cfg.max_agents,
                float(on_goal[ego, :count].float().mean()), float(finite_hops.mean()),
                float(finite_hops.std(unbiased=False)), float(finite_hops.min()), float(finite_hops.max()),
                float((history_selected[ego, :count, -1] == int(Action.WAIT)).float().mean()),
                float((history_selected[ego, :count, -1] != history_executed[ego, :count, -1]).float().mean()),
                float(csf[ego, :count].float().mean()), float(cgr[ego, :count].float().mean()),
                float((candidate_contenders[ego, :count] > 0).float().mean()),
                float(candidate_edge_swap[ego, :count].float().mean()),
                float(cbn[ego, :count].float().mean()), float(ccg[ego, :count].mean()),
                float(history_valid[ego, :count].float().mean()),
            ), dtype=torch.float32))
        scene = torch.stack(scene_rows)

        zeros_valid = torch.zeros_like(valid)
        batch = PolicyBatch(
            local_maps=raw["local_maps"].long(), agent_xy=raw["agent_xy"].long(),
            goal_delta=raw["goal_delta"].long(), remaining_hops=remaining, agent_valid=valid,
            track_reset=zeros_valid, on_goal=on_goal, goal_outside=raw["goal_outside"].bool(),
            history_selected=history_selected, history_executed=history_executed,
            history_outcome=history_outcome, history_delta_ctg=history_delta, history_valid=history_valid,
            candidate_target_xy=raw["candidate_target_xy"].long(), candidate_in_view=raw["candidate_in_view"].bool(),
            candidate_delta_ctg=raw["candidate_delta_ctg"].long(), candidate_greedy=cgr,
            candidate_static_free=csf, candidate_target_occupied=raw["candidate_target_occupied"].bool(),
            candidate_contenders=candidate_contenders, candidate_edge_swap=candidate_edge_swap,
            candidate_bottleneck=cbn, candidate_congestion=ccg,
            event_action=event_action, event_executed=event_executed, event_outcome=event_outcome,
            event_numeric=event_numeric, event_valid=event_valid, scene_numeric=scene,
            task_mode=torch.full((b,), self.task_mode_id), target_mode=torch.full((b,), self.target_mode_id),
        )
        return batch, raw["slot_ids"].long()

    def generate(self, obstacles: np.ndarray, positions: np.ndarray, goals: np.ndarray) -> PolicyBatch:
        batch, _ = self.generate_with_slot_ids(obstacles, positions, goals)
        return batch

    def commit_actions(self, actions: np.ndarray) -> None:
        if self._native is None:
            raise RuntimeError("generate must be called before commit_actions")
        self._native.commit_actions(np.ascontiguousarray(actions, dtype=np.int64))
