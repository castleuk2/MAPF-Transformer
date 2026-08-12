from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .communication import MultiRoundCommunicationPolicy
from .config import ModelConfig
from .cpp import CppEpisodeFeatureBuilder, load_extension
from .model import build_policy


class SSTPolicyRuntime:
    """Online all-Ego rollout adapter for rounds=0 or learned communication."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda:0", *, rounds: int = 0) -> None:
        self.device = torch.device(device)
        self.rounds = int(rounds)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        values = dict(payload["config"]["model"])
        project = Path(__file__).resolve().parents[2]
        values["map_checkpoint"] = str(
            project.parent
            / "mapf-structured-map-transformer/runs/policy_exposure_structured_25_ce/best.pt"
        )
        self.config = ModelConfig(**values)
        self.base = build_policy(self.config).to(self.device)
        load_checkpoint(checkpoint, model=self.base, map_location=self.device)
        self.model = MultiRoundCommunicationPolicy(self.base).to(self.device).eval()
        self.builder = CppEpisodeFeatureBuilder(self.config)
        self.reset()

    def reset(self) -> None:
        self._positions: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._native = None

    def act_global(
        self, obstacles: np.ndarray, positions: np.ndarray, goals: np.ndarray
    ) -> list[int]:
        positions = np.ascontiguousarray(positions, dtype=np.int64)
        self._positions.append(positions.copy())
        # The current row is a target placeholder only. History features access
        # rows strictly before the current time step.
        action_rows = self._actions + [np.zeros(positions.shape[0], dtype=np.int64)]
        position_history = np.ascontiguousarray(np.stack(self._positions), dtype=np.int64)
        action_history = np.ascontiguousarray(np.stack(action_rows), dtype=np.int64)
        if self._native is None:
            self._native = load_extension().EpisodeFeatureGenerator(
                np.ascontiguousarray(obstacles, dtype=np.uint8), position_history,
                np.ascontiguousarray(goals, dtype=np.int64), action_history,
                self.config.local_map_size, self.config.core_map_size,
                self.config.max_current_agents, self.config.history_tracks,
                self.config.history_steps, self.config.max_hops,
            )
        else:
            self._native.update_history(position_history, action_history)
        batch, graph = self.builder.convert_raw(
            self._native.build(len(self._positions) - 1)
        )
        batch, graph = batch.to(self.device), graph.to(self.device)
        with torch.inference_mode():
            if self.rounds > 0:
                logits = self.model(batch, graph, rounds=self.rounds).final.ego_logits
            else:
                logits = self.base(batch, coordination_mode="none").ego_logits
        actions = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        self._actions.append(actions)
        return actions.tolist()
