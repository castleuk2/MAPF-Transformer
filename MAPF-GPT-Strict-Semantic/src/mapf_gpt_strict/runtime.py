from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .continuous_tokenizer import EncodedObservation, StableSlotAllocator
from .expert_dataset import DELTAS, INF
from .model import GPTConfig, MAPFGPT
from .resolver import CSPIBTResolver


class StrictSemanticPolicy:
    """Central batched rollout adapter: one Ego sample per active agent."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = MAPFGPT(GPTConfig(**payload["config"])).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.position_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []
        self.distance_cache: dict[tuple[int, int], np.ndarray] = {}
        self.slot_allocators: list[StableSlotAllocator] = []
        self.last_slot_ids: list[tuple[object | None, ...]] = []
        self.wait_age: np.ndarray | None = None

    def _slot_ids(self, positions: np.ndarray, ego: int) -> tuple[object | None, ...]:
        if len(self.slot_allocators) != len(positions):
            self.slot_allocators = [StableSlotAllocator() for _ in range(len(positions))]
        ego_position = positions[ego]
        visible = [i for i in range(len(positions)) if np.abs(positions[i] - ego_position).max() <= 8]
        distance = {i: int(np.abs(positions[i] - ego_position).sum()) for i in visible}
        return self.slot_allocators[ego].assign_ids(ego, visible, distance)

    @staticmethod
    def _distance_map(obstacles: np.ndarray, goal: np.ndarray) -> np.ndarray:
        h, w = obstacles.shape
        distance = np.full((h, w), INF, dtype=np.int32)
        target = tuple(map(int, goal))
        if not (0 <= target[0] < h and 0 <= target[1] < w) or obstacles[target]:
            return distance
        queue = deque([target]); distance[target] = 0
        while queue:
            row, col = queue.popleft(); next_distance = int(distance[row, col]) + 1
            for dr, dc in DELTAS[1:]:
                nr, nc = row + int(dr), col + int(dc)
                if 0 <= nr < h and 0 <= nc < w and not obstacles[nr, nc] and distance[nr, nc] == INF:
                    distance[nr, nc] = next_distance; queue.append((nr, nc))
        return distance

    def _distance(self, obstacles: np.ndarray, goal: np.ndarray) -> np.ndarray:
        key = tuple(map(int, goal))
        if key not in self.distance_cache:
            self.distance_cache[key] = self._distance_map(obstacles, goal)
        return self.distance_cache[key]

    def _encode_ego(
        self, obstacles: np.ndarray, positions: np.ndarray, goals: np.ndarray, ego: int
    ) -> tuple[EncodedObservation, torch.Tensor]:
        features = np.zeros((256, 16), np.float32)
        agent = np.full(256, 13, np.int64); field = np.full(256, 10, np.int64)
        lag = np.full(256, 6, np.int64); role = np.zeros(256, np.int64)
        valid = np.zeros(256, np.int64); field[:25] = 0; valid[:25] = 1
        ego_position = positions[ego]
        slot_ids = self._slot_ids(positions, ego)
        visible = {i for i in range(len(positions)) if np.abs(positions[i] - ego_position).max() <= 8}
        slots = [gid if gid in visible else None for gid in slot_ids]
        h, w = obstacles.shape
        halo = np.ones((17, 17), np.int64); row0, col0 = ego_position - 8
        rs, re = max(0, row0), min(h, row0 + 17); cs, ce = max(0, col0), min(w, col0 + 17)
        halo[rs-row0:re-row0, cs-col0:ce-col0] = obstacles[rs:re, cs:ce]
        distance_maps: list[np.ndarray | None] = [None] * 13
        for slot, gid in enumerate(slots):
            if gid is None:
                continue
            distance = self._distance(obstacles, goals[gid]); distance_maps[slot] = distance
            base = 25 + slot * 8; relative = positions[gid] - ego_position
            goal_delta = goals[gid] - positions[gid]; hops = int(distance[tuple(positions[gid])])
            features[base, :2] = np.clip(relative, -8, 8) / 8
            features[base+1, :2] = np.clip(goal_delta, -8, 8) / 8
            features[base+2, :2] = (0, 1) if hops == INF else (min(hops, 32) / 32, 0)
            occupied = {tuple(x) for j, x in enumerate(positions) if j != gid}
            for action, (dr, dc) in enumerate(DELTAS):
                index = base + 3 + action; target = tuple(positions[gid] + (dr, dc))
                inside = 0 <= target[0] < h and 0 <= target[1] < w
                free = inside and not obstacles[target]; occupied_target = target in occupied
                target_hops = int(distance[target]) if free else INF
                delta = -1 if free and hops != INF and target_hops < hops else 1 if free and hops != INF and target_hops > hops else 0
                state = 2 if occupied_target else 0 if free else 1
                features[index, :5] = (float(free and not occupied_target), delta, state == 0, state == 1, state == 2)
            agent[base:base+8] = slot; field[base:base+8] = np.arange(1, 9)
            lag[base:base+8] = 0; role[base:base+8] = 1 if slot == 0 else 2; valid[base:base+8] = 1
        # Match training: current visible identities define the six history tracks.
        history_positions = self.position_history[-5:]
        history_actions = self.action_history[-5:]
        missing = 5 - len(history_positions)
        for slot, gid in enumerate(slot_ids[:6]):
            if gid is None:
                continue
            distance = self._distance(obstacles, goals[gid])
            for history_index, past_positions in enumerate(history_positions):
                step = missing + history_index; base = 129 + (slot * 5 + step) * 4
                relative = past_positions[gid] - ego_position
                goal_delta = goals[gid] - past_positions[gid]; hops = int(distance[tuple(past_positions[gid])])
                features[base, :2] = np.clip(relative, -8, 8) / 8
                features[base+1, :2] = np.clip(goal_delta, -8, 8) / 8
                features[base+2, :2] = (0, 1) if hops == INF else (min(hops, 32) / 32, 0)
                selected = int(history_actions[history_index][gid])
                next_positions = history_positions[history_index + 1] if history_index + 1 < len(history_positions) else positions
                movement = next_positions[gid] - past_positions[gid]
                observed = next((a for a, d in enumerate(DELTAS) if np.array_equal(movement, d)), 0)
                features[base+3, selected if 0 <= selected < 5 else 5] = 1
                features[base+3, 6 + (observed if 0 <= observed < 5 else 5)] = 1
                agent[base:base+4] = slot; field[base:base+4] = (1, 2, 3, 9)
                lag[base:base+4] = 5-step; role[base:base+4] = 1 if slot == 0 else 2; valid[base:base+4] = 1
        encoded = EncodedObservation(*[torch.from_numpy(x) for x in (features, agent, field, lag, role, valid)])
        return encoded, torch.from_numpy(halo)

    @torch.inference_mode()
    def preference_logits(self, obstacles: np.ndarray, positions: np.ndarray, goals: np.ndarray) -> torch.Tensor:
        obstacles = np.asarray(obstacles, dtype=np.uint8); positions = np.asarray(positions, dtype=np.int64)
        goals = np.asarray(goals, dtype=np.int64)
        samples = [self._encode_ego(obstacles, positions, goals, ego) for ego in range(len(positions))]
        self.last_slot_ids = [allocator.slot_to_id for allocator in self.slot_allocators]
        fields = []
        for name in EncodedObservation.__dataclass_fields__:
            fields.append(torch.stack([getattr(sample[0], name) for sample in samples]).to(self.device))
        encoded = EncodedObservation(*fields); halos = torch.stack([sample[1] for sample in samples]).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits, _ = self.model(encoded, halo_maps=halos)
        return logits.float().cpu()

    @torch.inference_mode()
    def act_global(self, obstacles: np.ndarray, positions: np.ndarray, goals: np.ndarray) -> list[int]:
        positions = np.asarray(positions, dtype=np.int64)
        logits = self.preference_logits(obstacles, positions, goals)
        selected = logits.argmax(dim=-1).numpy().astype(np.int64)
        self.position_history.append(positions.copy()); self.action_history.append(selected.copy())
        self.position_history = self.position_history[-5:]; self.action_history = self.action_history[-5:]
        return selected.tolist()


class StrictSemanticMPCTPolicy(StrictSemanticPolicy):
    """Strict Semantic tokens with centralized CS-PIBT preference coordination."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda:0") -> None:
        super().__init__(checkpoint, device)
        self.resolver = CSPIBTResolver()

    @torch.inference_mode()
    def act_global(self, obstacles: np.ndarray, positions: np.ndarray, goals: np.ndarray) -> list[int]:
        obstacles = np.asarray(obstacles, dtype=np.uint8)
        positions = np.asarray(positions, dtype=np.int64)
        goals = np.asarray(goals, dtype=np.int64)
        logits = self.preference_logits(obstacles, positions, goals)
        selected = logits.argmax(dim=-1).numpy().astype(np.int64)
        if self.wait_age is None or len(self.wait_age) != len(positions):
            self.wait_age = np.zeros(len(positions), dtype=np.float32)
        actions = self.resolver.resolve(
            positions,
            logits,
            obstacles,
            wait_age=self.wait_age,
            on_goal=np.all(positions == goals, axis=1),
        )
        self.wait_age = np.where(actions == 0, self.wait_age + 1, 0).astype(np.float32)
        self.position_history.append(positions.copy()); self.action_history.append(selected.copy())
        self.position_history = self.position_history[-5:]; self.action_history = self.action_history[-5:]
        return actions.tolist()
