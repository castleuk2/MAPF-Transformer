from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

from ..config import ModelConfig
from ..data.npz_dataset import Episode, EpisodeFeatureBuilder
from ..types import CommunicationGraph, PolicyBatch


_EXTENSION = None


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        source = Path(__file__).with_name("_sst_feature_generator.cpp")
        _EXTENSION = load(
            name="mapf_sst_feature_generator",
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17"],
            with_cuda=False,
            verbose=False,
        )
    return _EXTENSION


class CppEpisodeFeatureBuilder(EpisodeFeatureBuilder):
    """Exact all-Ego SST feature construction in one native call per frame."""

    def __init__(self, config: ModelConfig, *, coordinate_order: str = "row_col", cache_size: int = 4) -> None:
        super().__init__(config, coordinate_order=coordinate_order, cache_size=cache_size)
        self._module = load_extension()
        self._native: OrderedDict[int, tuple[Episode, object]] = OrderedDict()

    def _native_episode(self, episode: Episode):
        key = id(episode)
        if key not in self._native:
            native = self._module.EpisodeFeatureGenerator(
                np.ascontiguousarray(episode.obstacles, dtype=np.uint8),
                np.ascontiguousarray(episode.positions, dtype=np.int64),
                np.ascontiguousarray(episode.goals, dtype=np.int64),
                np.ascontiguousarray(episode.actions, dtype=np.int64),
                self.config.local_map_size,
                self.config.core_map_size,
                self.config.max_current_agents,
                self.config.history_tracks,
                self.config.history_steps,
                self.config.message_neighbors,
                self.config.max_hops,
            )
            self._native[key] = (episode, native)
        self._native.move_to_end(key)
        while len(self._native) > self.cache_size:
            self._native.popitem(last=False)
        return self._native[key][1]

    @staticmethod
    def _tensor(raw: dict, name: str, *, boolean: bool = False) -> torch.Tensor:
        value = torch.from_numpy(np.asarray(raw[name]))
        return value.bool() if boolean else value.long()

    def build_all_views(self, episode: Episode, time_step: int) -> tuple[PolicyBatch, CommunicationGraph]:
        raw = self._native_episode(episode).build(int(time_step))
        return self.convert_raw(raw)

    def convert_raw(self, raw) -> tuple[PolicyBatch, CommunicationGraph]:
        t = self._tensor
        batch = PolicyBatch(
            local_maps=t(raw, "local_maps"),
            current_xy=t(raw, "current_xy"),
            current_goal_delta=t(raw, "current_goal_delta"),
            current_hops=t(raw, "current_hops"),
            current_valid=t(raw, "current_valid", boolean=True),
            current_track_reset=t(raw, "current_track_reset", boolean=True),
            current_global_ids=t(raw, "current_global_ids"),
            candidate_target_core_xy=t(raw, "candidate_target_core_xy"),
            candidate_in_core=t(raw, "candidate_in_core", boolean=True),
            candidate_static_free=t(raw, "candidate_static_free", boolean=True),
            candidate_one_hop_hops=t(raw, "candidate_one_hop_hops"),
            candidate_delta_ctg=t(raw, "candidate_delta_ctg"),
            candidate_greedy=t(raw, "candidate_greedy", boolean=True),
            candidate_bottleneck=t(raw, "candidate_bottleneck", boolean=True),
            candidate_dynamic_occupied=t(raw, "candidate_dynamic_occupied", boolean=True),
            history_xy=t(raw, "history_xy"),
            history_goal_delta=t(raw, "history_goal_delta"),
            history_hops=t(raw, "history_hops"),
            history_selected_action=t(raw, "history_selected_action"),
            history_observed_move=t(raw, "history_observed_move"),
            history_valid=t(raw, "history_valid", boolean=True),
            history_track_current_slot=t(raw, "history_track_current_slot"),
            history_global_ids=t(raw, "history_global_ids"),
            ego_action=t(raw, "ego_action"),
            action_soft_targets=None,
        )
        graph = CommunicationGraph(
            neighbor_view_index=t(raw, "neighbor_view_index"),
            neighbor_valid=t(raw, "neighbor_valid", boolean=True),
            neighbor_current_slot=t(raw, "neighbor_current_slot"),
        )
        return batch, graph
