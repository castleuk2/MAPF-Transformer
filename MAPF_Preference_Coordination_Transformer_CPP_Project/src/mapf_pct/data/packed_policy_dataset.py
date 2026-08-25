from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ModelConfig
from ..types import PolicyBatch


PACKED_POLICY_VERSION = 2


def model_fingerprint(config: ModelConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class _EpisodeCache:
    def __init__(self, size: int = 4) -> None:
        self.size = int(size)
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> dict[str, np.ndarray]:
        value = self.values.pop(path, None)
        if value is None:
            with np.load(path, allow_pickle=False) as archive:
                value = {name: np.asarray(archive[name]) for name in archive.files}
        self.values[path] = value
        while len(self.values) > self.size:
            self.values.popitem(last=False)
        return value


class PackedPolicyDataset(Dataset[PolicyBatch]):
    """Restore precomputed PolicyBatch tensors with exact dtype/value parity."""

    def __init__(self, config: ModelConfig, manifest: str | Path, cache_size: int = 4) -> None:
        manifest = Path(manifest).resolve()
        self.records: list[dict] = []
        counts: list[int] = []
        expected_fingerprint = model_fingerprint(config)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("packed_policy_version", -1)) != PACKED_POLICY_VERSION:
                raise ValueError(f"unsupported packed policy version: {record}")
            if record.get("model_fingerprint") != expected_fingerprint:
                raise ValueError(
                    "packed features were generated with a different ModelConfig: "
                    f"{record.get('model_fingerprint')} != {expected_fingerprint}"
                )
            path = Path(record["path"])
            if not path.is_absolute():
                path = manifest.parent / path
            record["path"] = str(path.resolve())
            record["sample_counts"] = np.asarray(record["sample_counts"], dtype=np.int64)
            self.records.append(record)
            counts.append(int(record["samples"]))
        if not self.records:
            raise ValueError(f"empty packed manifest: {manifest}")
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        self.length = int(self.cumulative[-1])
        self.cache = _EpisodeCache(cache_size)
        self.config = config

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> PolicyBatch:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        episode = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = int(self.cumulative[episode - 1]) if episode else 0
        local = int(index - previous)
        arrays = self.cache.get(self.records[episode]["path"])
        values = {}
        for field in fields(PolicyBatch):
            if field.name not in arrays:
                values[field.name] = None
                continue
            tensor = torch.from_numpy(np.array(arrays[field.name][local], copy=True))
            dtype_name = str(arrays[f"__dtype__{field.name}"].item())
            values[field.name] = tensor.to(getattr(torch, dtype_name))
        return PolicyBatch(**values)
