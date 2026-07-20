from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .dataset import EpisodeSequenceDataset
from .features import (
    GoalDistanceCache,
    START_ACTION,
    TransitionOutcome,
    build_frame_features,
    visible_neighbor_ids,
)
from .geometry import CTG_UNREACHABLE
from .tracking import StableNeighborTracker


PACKED_FORMAT_VERSION = 1


def pack_local_map(local_map: np.ndarray) -> np.ndarray:
    """Packs a binary local map in row-major, little-bit order."""
    values = np.asarray(local_map, dtype=np.uint8).reshape(-1)
    if np.any(values > 1):
        raise ValueError("local maps must be binary before bit packing")
    return np.packbits(values, bitorder="little")


def pack_agent_fields(
    x: np.ndarray,
    y: np.ndarray,
    action_mask: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Packs x(4), y(4), shortest-path mask(4), and distance(6)."""
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    distance = np.asarray(distance, dtype=np.uint32)
    mask = np.asarray(action_mask, dtype=np.uint8)
    if mask.shape != x.shape + (4,):
        raise ValueError("action_mask must have shape x.shape + (4,)")
    mask_bits = sum(mask[..., bit].astype(np.uint32) << bit for bit in range(4))
    return x | (y << 4) | (mask_bits << 8) | (distance << 12)


def pack_boolean_slots(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if values.shape[-1] > 16:
        raise ValueError("slot bit masks support at most 16 slots")
    result = np.zeros(values.shape[:-1], dtype=np.uint16)
    for bit in range(values.shape[-1]):
        result |= values[..., bit].astype(np.uint16) << bit
    return result


def _tracking_for_episode(episode: Any, config: ModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = episode.time_steps
    agents = episode.num_agents
    if episode.neighbor_ids is not None:
        ids = np.asarray(episode.neighbor_ids[:frames], dtype=np.int16)
        valid = (
            np.asarray(episode.neighbor_valid[:frames], dtype=bool)
            if episode.neighbor_valid is not None
            else ids >= 0
        )
        reset = (
            np.asarray(episode.track_reset[:frames], dtype=bool)
            if episode.track_reset is not None
            else np.zeros_like(valid)
        )
        return ids, valid, reset

    ids = np.full((frames, agents, config.max_neighbors), -1, dtype=np.int16)
    valid = np.zeros_like(ids, dtype=bool)
    reset = np.zeros_like(ids, dtype=bool)
    for ego_id in range(agents):
        tracker = StableNeighborTracker(config.max_neighbors, grace_steps=1)
        for frame in range(frames):
            visible, scores = visible_neighbor_ids(
                episode.positions[frame], ego_id, config.local_radius
            )
            tracked = tracker.update(visible, scores)
            ids[frame, ego_id] = tracked.agent_ids
            valid[frame, ego_id] = tracked.valid
            reset[frame, ego_id] = tracked.reset
    return ids, valid, reset


def build_packed_episode(episode: Any, config: ModelConfig) -> dict[str, np.ndarray]:
    """Precomputes each real (frame, ego) feature once in compact form."""
    config.validate()
    frames, agents = episode.time_steps, episode.num_agents
    map_bytes = (config.map_size * config.map_size + 7) // 8
    local_map_bits = np.empty((frames, agents, map_bytes), dtype=np.uint8)
    agent_payload = np.empty(
        (frames, agents, config.agents_per_frame), dtype=np.uint32
    )
    agent_valid_bits = np.empty((frames, agents), dtype=np.uint16)
    track_reset_bits = np.empty((frames, agents), dtype=np.uint16)
    previous_action = np.empty((frames, agents), dtype=np.uint8)
    actual_move = np.empty((frames, agents), dtype=np.uint8)
    outcome = np.empty((frames, agents), dtype=np.uint8)
    visible_count = np.empty((frames, agents), dtype=np.uint8)
    one_hop_ctg = (
        np.empty(
            (frames, agents, config.agents_per_frame, config.num_actions),
            dtype=np.uint8,
        )
        if config.one_hop_ctg
        else None
    )

    neighbor_ids, neighbor_valid, track_reset = _tracking_for_episode(episode, config)
    distance_cache = GoalDistanceCache(episode.obstacles)
    for frame in range(frames):
        previous_positions = episode.positions[frame - 1] if frame > 0 else None
        for ego_id in range(agents):
            features = build_frame_features(
                obstacles=episode.obstacles,
                positions=episode.positions[frame],
                goals=episode.goals_at(frame),
                ego_id=ego_id,
                config=config,
                distance_cache=distance_cache,
                neighbor_slot_ids=neighbor_ids[frame, ego_id],
                neighbor_valid=neighbor_valid[frame, ego_id],
                neighbor_reset=track_reset[frame, ego_id],
                previous_positions=previous_positions,
                previous_action=(
                    int(episode.actions[frame - 1, ego_id]) if frame > 0 else None
                ),
            )
            local_map_bits[frame, ego_id] = pack_local_map(features.local_map)
            agent_payload[frame, ego_id] = pack_agent_fields(
                features.agent_x,
                features.agent_y,
                features.action_mask,
                features.distance,
            )
            agent_valid_bits[frame, ego_id] = pack_boolean_slots(features.agent_valid)
            track_reset_bits[frame, ego_id] = pack_boolean_slots(features.track_reset)
            previous_action[frame, ego_id] = features.previous_action
            actual_move[frame, ego_id] = features.actual_move
            outcome[frame, ego_id] = features.outcome
            visible_count[frame, ego_id] = features.visible_count
            if one_hop_ctg is not None:
                one_hop_ctg[frame, ego_id] = features.one_hop_ctg.astype(np.uint8)

    arrays: dict[str, np.ndarray] = {
        "local_map_bits": local_map_bits,
        "agent_payload": agent_payload,
        "agent_valid_bits": agent_valid_bits,
        "track_reset_bits": track_reset_bits,
        "previous_action": previous_action,
        "actual_move": actual_move,
        "outcome": outcome,
        "visible_count": visible_count,
        "actions": np.asarray(episode.actions, dtype=np.uint8),
        "arrival_steps": np.asarray(episode.get_arrival_steps(), dtype=np.int32),
    }
    if one_hop_ctg is not None:
        arrays["one_hop_ctg"] = one_hop_ctg
    return arrays


class _PackedEpisodeLRU:
    def __init__(self, max_items: int = 4) -> None:
        self.max_items = int(max_items)
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> dict[str, np.ndarray]:
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        with np.load(path, allow_pickle=False) as data:
            value = {name: np.asarray(data[name]) for name in data.files}
        self.cache[path] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


class PackedEpisodeSequenceDataset(EpisodeSequenceDataset):
    """Builds temporal samples by indexing precomputed packed frame features."""

    def __init__(self, *args: Any, episode_cache_size: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, episode_cache_size=episode_cache_size, **kwargs)
        self._packed_cache = _PackedEpisodeLRU(episode_cache_size)
        for record in self.records:
            metadata = record.get("packed_format", {})
            if int(metadata.get("version", -1)) != PACKED_FORMAT_VERSION:
                raise ValueError(f"Unsupported packed cache format in {record['path']}")
            expected = {
                "map_size": self.config.map_size,
                "max_neighbors": self.config.max_neighbors,
                "one_hop_ctg": self.config.one_hop_ctg,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"Packed cache {key}={metadata.get(key)!r} does not match model {value!r}"
                    )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, time_step, ego_id = self._resolve_index(index)
        record = self.records[episode_index]
        episode = self._packed_cache.get(record["path"])
        first_frame = max(0, time_step - self.config.history_frames + 1)
        valid_frames = list(range(first_frame, time_step + 1))
        if self.history_augmentation:
            available = len(valid_frames)
            low = min(self.min_history_frames, available)
            rng = np.random.default_rng(self.seed + index * 104729)
            keep = int(rng.integers(low, available + 1))
            valid_frames = valid_frames[-keep:]

        frame_count = self.config.history_frames
        pad = frame_count - len(valid_frames)
        map_bytes = (self.config.map_size * self.config.map_size + 7) // 8
        slots = self.config.agents_per_frame
        empty_payload = np.uint32(15 | (15 << 4) | (63 << 12))

        local_map_bits = np.zeros((frame_count, map_bytes), dtype=np.uint8)
        agent_payload = np.full((frame_count, slots), empty_payload, dtype=np.uint32)
        agent_valid_bits = np.zeros(frame_count, dtype=np.uint16)
        track_reset_bits = np.zeros(frame_count, dtype=np.uint16)
        previous_action = np.full(frame_count, START_ACTION, dtype=np.uint8)
        actual_move = np.zeros(frame_count, dtype=np.uint8)
        outcome = np.full(frame_count, int(TransitionOutcome.START), dtype=np.uint8)
        visible_count = np.zeros(frame_count, dtype=np.uint8)
        frame_valid = np.zeros(frame_count, dtype=bool)
        one_hop_ctg = (
            np.full(
                (frame_count, slots, self.config.num_actions),
                CTG_UNREACHABLE,
                dtype=np.uint8,
            )
            if self.config.one_hop_ctg
            else None
        )

        if valid_frames:
            source = np.asarray(valid_frames, dtype=np.int64)
            target = slice(pad, frame_count)
            local_map_bits[target] = episode["local_map_bits"][source, ego_id]
            agent_payload[target] = episode["agent_payload"][source, ego_id]
            agent_valid_bits[target] = episode["agent_valid_bits"][source, ego_id]
            track_reset_bits[target] = episode["track_reset_bits"][source, ego_id]
            previous_action[target] = episode["previous_action"][source, ego_id]
            actual_move[target] = episode["actual_move"][source, ego_id]
            outcome[target] = episode["outcome"][source, ego_id]
            visible_count[target] = episode["visible_count"][source, ego_id]
            frame_valid[target] = True
            if one_hop_ctg is not None:
                one_hop_ctg[target] = episode["one_hop_ctg"][source, ego_id]

        sample = {
            "packed_local_maps": torch.from_numpy(local_map_bits),
            # All payloads are below 2^18, so signed int32 preserves every bit
            # while retaining broad PyTorch CPU/CUDA transfer support.
            "packed_agent_payload": torch.from_numpy(agent_payload.astype(np.int32)),
            "packed_agent_valid": torch.from_numpy(agent_valid_bits.astype(np.int32)),
            "packed_track_reset": torch.from_numpy(track_reset_bits.astype(np.int32)),
            "previous_action": torch.from_numpy(previous_action),
            "actual_move": torch.from_numpy(actual_move),
            "outcome": torch.from_numpy(outcome),
            "visible_count": torch.from_numpy(visible_count),
            "frame_valid": torch.from_numpy(frame_valid),
            "target": torch.as_tensor(int(episode["actions"][time_step, ego_id]), dtype=torch.long),
            "ego_id": torch.as_tensor(ego_id, dtype=torch.long),
            "time_step": torch.as_tensor(time_step, dtype=torch.long),
        }
        if one_hop_ctg is not None:
            sample["packed_one_hop_ctg"] = torch.from_numpy(one_hop_ctg)
        return sample


def unpack_packed_batch(
    batch: dict[str, torch.Tensor], config: ModelConfig
) -> dict[str, torch.Tensor]:
    """Losslessly restores the existing model input tensors on their current device."""
    if "packed_local_maps" not in batch:
        return batch
    packed_maps = batch.pop("packed_local_maps").to(torch.uint8)
    bit_ids = torch.arange(8, device=packed_maps.device, dtype=torch.uint8)
    map_values = ((packed_maps.unsqueeze(-1) >> bit_ids) & 1).flatten(-2)
    cells = config.map_size * config.map_size
    batch["local_maps"] = map_values[..., :cells].reshape(
        *packed_maps.shape[:-1], config.map_size, config.map_size
    ).long()

    payload = batch.pop("packed_agent_payload").long()
    batch["agent_x"] = payload & 0xF
    batch["agent_y"] = (payload >> 4) & 0xF
    mask_bits = (payload >> 8) & 0xF
    direction_bits = torch.arange(4, device=payload.device, dtype=torch.long)
    batch["action_mask"] = ((mask_bits.unsqueeze(-1) >> direction_bits) & 1).float()
    batch["distance"] = (payload >> 12) & 0x3F

    slot_bits = torch.arange(
        config.agents_per_frame, device=payload.device, dtype=torch.long
    )
    valid_bits = batch.pop("packed_agent_valid").long()
    reset_bits = batch.pop("packed_track_reset").long()
    batch["agent_valid"] = ((valid_bits.unsqueeze(-1) >> slot_bits) & 1).bool()
    batch["track_reset"] = ((reset_bits.unsqueeze(-1) >> slot_bits) & 1).bool()
    for name in ("previous_action", "actual_move", "outcome", "visible_count"):
        batch[name] = batch[name].long()
    if config.one_hop_ctg:
        batch["one_hop_ctg"] = batch.pop("packed_one_hop_ctg").long()
    return batch


def packed_manifest_record(
    output_path: Path,
    manifest_path: Path,
    source_record: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    try:
        stored_path = output_path.relative_to(manifest_path.parent)
    except ValueError:
        stored_path = output_path
    return {
        "path": str(stored_path),
        "source_path": source_record["path"],
        "time_steps": int(source_record["time_steps"]),
        "num_agents": int(source_record["num_agents"]),
        "arrival_steps": list(source_record["arrival_steps"]),
        "num_samples": int(source_record.get("num_samples", sum(source_record["arrival_steps"]))),
        "packed_format": {
            "version": PACKED_FORMAT_VERSION,
            "map_size": config.map_size,
            "max_neighbors": config.max_neighbors,
            "one_hop_ctg": config.one_hop_ctg,
        },
    }


def write_packed_manifest(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
