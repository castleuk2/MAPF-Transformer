from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from mapf_sst.config import load_config
from mapf_sst.cpp import CppEpisodeFeatureBuilder
from mapf_sst.data.npz_dataset import EpisodeFeatureBuilder, _read_manifest
from mapf_sst.model import SemanticSpatiotemporalPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for label, manifest in (("train", config.data.train_manifest), ("validation", config.data.val_manifest)):
        if not manifest or not Path(manifest).resolve().is_file():
            raise FileNotFoundError(f"{label} manifest missing: {manifest}")
        paths = _read_manifest(manifest)
        missing = sum(not path.is_file() for path in paths)
        if missing:
            raise FileNotFoundError(f"{label}: {missing} episode files are missing")
        print(f"{label}: episodes={len(paths)}, missing=0")

    model = SemanticSpatiotemporalPolicy(config.model)
    if config.model.freeze_map_encoder and not model.map_encoder.is_frozen:
        raise RuntimeError("Map encoder freeze contract failed")
    print("map checkpoint/load/freeze: OK")

    python_builder = EpisodeFeatureBuilder(config.model, coordinate_order=config.data.coordinate_order)
    cpp_builder = CppEpisodeFeatureBuilder(config.model, coordinate_order=config.data.coordinate_order)
    path = _read_manifest(config.data.train_manifest)[0]
    py_episode, cpp_episode = python_builder.load_episode(path), cpp_builder.load_episode(path)
    step = min(3, py_episode.actions.shape[0] - 1)
    expected, expected_graph = python_builder.build_all_views(py_episode, step)
    actual, actual_graph = cpp_builder.build_all_views(cpp_episode, step)
    for field in fields(expected):
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if left is not None and not torch.equal(left, right):
            raise RuntimeError(f"C++ feature mismatch: {field.name}")
    for name in ("neighbor_view_index", "neighbor_valid", "neighbor_current_slot"):
        if not torch.equal(getattr(expected_graph, name), getattr(actual_graph, name)):
            raise RuntimeError(f"C++ graph mismatch: {name}")
    print("C++ all-Ego feature/graph equality: OK")
    print("portable setup: OK")


if __name__ == "__main__":
    main()
