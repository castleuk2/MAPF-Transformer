from __future__ import annotations

import hashlib
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .checkpoint import load_model_from_checkpoint
from .config import ModelConfig, load_experiment_config
from .features import GoalDistanceCache, build_frame_features
from .model import MAPFTransformer
from .spatial_memory import EgoSpatialMemory
from .tracking import StableNeighborTracker
from .training import choose_device


@dataclass(slots=True)
class MAPFTransformerInferenceConfig:
    """Inference interface intentionally parallels MAPF-GPT's policy wrapper."""

    checkpoint_path: str | None = None
    model_config_path: str | None = None
    device: str = "auto"
    sample_actions: bool = False
    temperature: float = 1.0
    seed: int = 42
    strict_checkpoint: bool = True


@dataclass(slots=True)
class _EgoRuntimeState:
    memory: EgoSpatialMemory
    tracker: StableNeighborTracker
    encoded_frames: deque[torch.Tensor]
    encoded_token_valid: deque[torch.Tensor]
    map_latents: torch.Tensor | None = None


@dataclass(slots=True)
class _EnvironmentRuntimeState:
    obstacle_signature: str
    obstacles: np.ndarray
    ego_states: list[_EgoRuntimeState]
    distance_cache: GoalDistanceCache
    previous_positions: np.ndarray | None = None
    last_actions: np.ndarray | None = None
    step: int = 0


class MAPFTransformerInference:
    """Stateful decentralized policy wrapper for POGEMA-like MAPF observations.

    Required global observation fields are:
    ``global_obstacles``, ``global_xy`` and ``global_target_xy``. Current POGEMA
    supplies one coordinate/goal per agent dictionary; the wrapper stacks them.
    The returned actions use POGEMA order: WAIT, UP, DOWN, LEFT, RIGHT.
    """

    def __init__(
        self,
        config: MAPFTransformerInferenceConfig | None = None,
        model: MAPFTransformer | None = None,
    ) -> None:
        self.inference_config = config or MAPFTransformerInferenceConfig()
        self.device = choose_device(self.inference_config.device)
        torch.manual_seed(self.inference_config.seed)
        self.generator: torch.Generator | None
        if self.device.type in {"cpu", "cuda"}:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.inference_config.seed)
        else:
            self.generator = None

        if model is not None:
            self.model = model.to(self.device).eval()
        elif self.inference_config.checkpoint_path:
            self.model, _ = load_model_from_checkpoint(
                self.inference_config.checkpoint_path,
                device=self.device,
                strict=self.inference_config.strict_checkpoint,
            )
        else:
            if self.inference_config.model_config_path:
                model_config = load_experiment_config(self.inference_config.model_config_path).model
            else:
                model_config = ModelConfig()
            self.model = MAPFTransformer(model_config).to(self.device).eval()
            warnings.warn(
                "No checkpoint was supplied; MAPFTransformerInference uses random weights.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.model_config = self.model.config
        self._states: dict[Any, _EnvironmentRuntimeState] = {}

    @staticmethod
    def _extract_global_fields(
        observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(observations, Mapping):
            observation_list = [observations]
        else:
            if len(observations) == 0:
                raise ValueError("observations must not be empty")
            observation_list = list(observations)
        first = observation_list[0]
        required = ("global_obstacles", "global_xy", "global_target_xy")
        missing = [field for field in required if field not in first]
        if missing:
            raise KeyError(
                "MAPF observations are missing global fields: " + ", ".join(missing)
            )
        obstacles = np.asarray(first["global_obstacles"], dtype=np.uint8)
        first_position = np.asarray(first["global_xy"], dtype=np.int16)
        first_goal = np.asarray(first["global_target_xy"], dtype=np.int16)
        # POGEMA 2.0 stores one global coordinate per agent dictionary. Older
        # wrappers may duplicate the full [N,2] arrays in every dictionary.
        positions = (
            np.stack([np.asarray(obs["global_xy"], dtype=np.int16) for obs in observation_list])
            if first_position.ndim == 1
            else first_position
        )
        goals = (
            np.stack([np.asarray(obs["global_target_xy"], dtype=np.int16) for obs in observation_list])
            if first_goal.ndim == 1
            else first_goal
        )
        if obstacles.ndim != 2:
            raise ValueError("global_obstacles must be 2-D")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("global_xy must have shape [N,2]")
        if goals.shape != positions.shape:
            raise ValueError("global_target_xy must have shape [N,2]")
        return obstacles, positions, goals

    @staticmethod
    def _obstacle_signature(obstacles: np.ndarray) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray(obstacles.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(obstacles).tobytes())
        return digest.hexdigest()

    def _new_state(self, obstacles: np.ndarray, num_agents: int) -> _EnvironmentRuntimeState:
        ego_states = [
            _EgoRuntimeState(
                memory=EgoSpatialMemory(self.model_config.map_size),
                tracker=StableNeighborTracker(self.model_config.max_neighbors, grace_steps=1),
                encoded_frames=deque(maxlen=self.model_config.history_frames),
                encoded_token_valid=deque(maxlen=self.model_config.history_frames),
            )
            for _ in range(num_agents)
        ]
        return _EnvironmentRuntimeState(
            obstacle_signature=self._obstacle_signature(obstacles),
            obstacles=np.asarray(obstacles, dtype=np.uint8).copy(),
            ego_states=ego_states,
            distance_cache=GoalDistanceCache(obstacles),
            last_actions=np.zeros(num_agents, dtype=np.int64),
        )

    def _ensure_state(
        self,
        environment_key: Any,
        obstacles: np.ndarray,
        num_agents: int,
    ) -> _EnvironmentRuntimeState:
        state = self._states.get(environment_key)
        signature = self._obstacle_signature(obstacles)
        if (
            state is None
            or state.obstacle_signature != signature
            or len(state.ego_states) != num_agents
        ):
            state = self._new_state(obstacles, num_agents)
            self._states[environment_key] = state
        return state

    @staticmethod
    def _frame_feature_tensors(features: Any, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "agent_x": torch.as_tensor(features.agent_x, dtype=torch.long, device=device).unsqueeze(0),
            "agent_y": torch.as_tensor(features.agent_y, dtype=torch.long, device=device).unsqueeze(0),
            "action_mask": torch.as_tensor(
                features.action_mask, dtype=torch.float32, device=device
            ).unsqueeze(0),
            "distance": torch.as_tensor(features.distance, dtype=torch.long, device=device).unsqueeze(0),
            "one_hop_ctg": torch.as_tensor(
                features.one_hop_ctg, dtype=torch.long, device=device
            ).unsqueeze(0),
            "agent_valid": torch.as_tensor(
                features.agent_valid, dtype=torch.bool, device=device
            ).unsqueeze(0),
            "track_reset": torch.as_tensor(
                features.track_reset, dtype=torch.bool, device=device
            ).unsqueeze(0),
            "previous_action": torch.as_tensor(
                [features.previous_action], dtype=torch.long, device=device
            ),
            "actual_move": torch.as_tensor([features.actual_move], dtype=torch.long, device=device),
            "outcome": torch.as_tensor([features.outcome], dtype=torch.long, device=device),
            "visible_count": torch.as_tensor(
                [features.visible_count], dtype=torch.long, device=device
            ),
        }

    @torch.no_grad()
    def _act_one(
        self,
        observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        environment_key: Any,
    ) -> list[int]:
        obstacles, positions, goals = self._extract_global_fields(observations)
        state = self._ensure_state(environment_key, obstacles, positions.shape[0])
        previous_positions = state.previous_positions
        previous_actions = state.last_actions.copy() if state.last_actions is not None else None

        # Update all persistent maps first, then encode only changed maps as one batch.
        changed_ego_ids: list[int] = []
        changed_maps: list[np.ndarray] = []
        for ego_id, ego_state in enumerate(state.ego_states):
            update = ego_state.memory.update(obstacles, positions[ego_id])
            if ego_state.map_latents is None or not update.reused:
                changed_ego_ids.append(ego_id)
                changed_maps.append(ego_state.memory.snapshot())
        if changed_maps:
            map_batch = torch.as_tensor(
                np.stack(changed_maps), dtype=torch.long, device=self.device
            )
            encoded_maps, _ = self.model.encode_maps(map_batch)
            for batch_index, ego_id in enumerate(changed_ego_ids):
                state.ego_states[ego_id].map_latents = encoded_maps[batch_index : batch_index + 1]

        features_list = []
        for ego_id, ego_state in enumerate(state.ego_states):
            features_list.append(
                build_frame_features(
                    obstacles=obstacles,
                    positions=positions,
                    goals=goals,
                    ego_id=ego_id,
                    config=self.model_config,
                    distance_cache=state.distance_cache,
                    tracker=ego_state.tracker,
                    previous_positions=previous_positions,
                    previous_action=(
                        int(previous_actions[ego_id])
                        if previous_positions is not None and previous_actions is not None
                        else None
                    ),
                    local_map=ego_state.memory.snapshot(),
                )
            )

        def stack_feature(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(
                np.stack([getattr(features, name) for features in features_list]),
                dtype=dtype,
                device=self.device,
            )

        map_latents = torch.cat(
            [ego_state.map_latents for ego_state in state.ego_states if ego_state.map_latents is not None],
            dim=0,
        )
        if map_latents.shape[0] != positions.shape[0]:
            raise RuntimeError("Not every Ego spatial memory produced map latents")
        frame_batch_current, token_valid_current = self.model.encode_frame_from_latents(
            map_latents=map_latents,
            agent_x=stack_feature("agent_x", torch.long),
            agent_y=stack_feature("agent_y", torch.long),
            distance=stack_feature("distance", torch.long),
            one_hop_ctg=stack_feature("one_hop_ctg", torch.long),
            agent_valid=stack_feature("agent_valid", torch.bool),
            track_reset=stack_feature("track_reset", torch.bool),
            previous_action=torch.as_tensor(
                [features.previous_action for features in features_list],
                dtype=torch.long,
                device=self.device,
            ),
            actual_move=torch.as_tensor(
                [features.actual_move for features in features_list],
                dtype=torch.long,
                device=self.device,
            ),
            outcome=torch.as_tensor(
                [features.outcome for features in features_list],
                dtype=torch.long,
                device=self.device,
            ),
            visible_count=torch.as_tensor(
                [features.visible_count for features in features_list],
                dtype=torch.long,
                device=self.device,
            ),
        )
        for ego_id, ego_state in enumerate(state.ego_states):
            ego_state.encoded_frames.append(frame_batch_current[ego_id].detach())
            ego_state.encoded_token_valid.append(token_valid_current[ego_id].detach())

        b = positions.shape[0]
        f = self.model_config.history_frames
        p = self.model_config.tokens_per_frame
        d = self.model_config.d_model
        frame_batch = torch.zeros((b, f, p, d), dtype=torch.float32, device=self.device)
        valid_batch = torch.zeros((b, f), dtype=torch.bool, device=self.device)
        token_valid_batch = torch.zeros((b, f, p), dtype=torch.bool, device=self.device)
        for ego_id, ego_state in enumerate(state.ego_states):
            count = len(ego_state.encoded_frames)
            start_index = f - count
            if count:
                frame_batch[ego_id, start_index:] = torch.stack(
                    list(ego_state.encoded_frames), dim=0
                )
                valid_batch[ego_id, start_index:] = True
                token_valid_batch[ego_id, start_index:] = torch.stack(
                    list(ego_state.encoded_token_valid), dim=0
                )

        output = self.model.forward_encoded_frames(
            frame_batch, valid_batch, frame_token_valid=token_valid_batch
        )
        temperature = max(float(self.inference_config.temperature), 1e-6)
        if self.inference_config.sample_actions:
            probabilities = torch.softmax(output.logits / temperature, dim=-1)
            actions = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=self.generator,
            ).squeeze(-1)
        else:
            actions = output.logits.argmax(dim=-1)

        action_list = [int(value) for value in actions.cpu().tolist()]
        state.previous_positions = positions.copy()
        state.last_actions = np.asarray(action_list, dtype=np.int64)
        state.step += 1
        return action_list

    def act(
        self,
        observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    ) -> list[int]:
        """Returns one action per agent for one environment."""
        return self._act_one(observations, environment_key=0)

    def act_batch(
        self,
        observations_batch: Sequence[Sequence[Mapping[str, Any]] | Mapping[str, Any]],
        positions: Any | None = None,
    ) -> list[list[int]]:
        """MAPF-GPT-like batched interface; ``positions`` is accepted for compatibility."""
        environment_keys = list(range(len(observations_batch))) if positions is None else list(positions)
        if len(environment_keys) != len(observations_batch):
            raise ValueError("positions/environment keys must match observations_batch length")
        return [
            self._act_one(observations, environment_key=environment_key)
            for environment_key, observations in zip(environment_keys, observations_batch)
        ]

    def reset_states(self, environment_index: Any | None = None) -> None:
        if environment_index is None:
            self._states.clear()
            return
        self._states.pop(environment_index, None)
