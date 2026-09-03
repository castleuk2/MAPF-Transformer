from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ModelConfig
from ..types import PolicyBatch


class HybridSelectedPolicyDataset(Dataset[PolicyBatch]):
    """Losslessly load an indexed view of precomputed Packed PolicyBatch files."""

    def __init__(self, config: ModelConfig, manifest: str | Path, cache_size: int = 4):
        del config
        manifest = Path(manifest).resolve()
        self.records: list[dict] = []
        counts: list[int] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("mode") != "packed_indexed":
                raise ValueError(f"unsupported selected-data mode: {record.get('mode')}")
            for key in ("packed_path", "index_path"):
                path = Path(record[key])
                if not path.is_absolute():
                    path = manifest.parent / path
                record[key] = str(path.resolve())
            self.records.append(record)
            counts.append(int(record["samples"]))
        if not self.records:
            raise ValueError(f"empty selected manifest: {manifest}")
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        self.length = int(self.cumulative[-1])
        self.cache_size = int(cache_size)
        self._arrays: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._indices: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.length

    def _index(self, path: str) -> np.ndarray:
        if path not in self._indices:
            with np.load(path, allow_pickle=False) as archive:
                self._indices[path] = np.asarray(archive["index"], dtype=np.int64)
        return self._indices[path]

    def __getitem__(self, index: int) -> PolicyBatch:
        if index < 0:
            index += self.length
        episode = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = int(self.cumulative[episode - 1]) if episode else 0
        record = self.records[episode]
        packed_index = int(self._index(record["index_path"])[index - previous])
        path = record["packed_path"]
        arrays = self._arrays.pop(path, None)
        if arrays is None:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
        self._arrays[path] = arrays
        while len(self._arrays) > self.cache_size:
            self._arrays.popitem(last=False)
        values = {}
        for field in fields(PolicyBatch):
            if field.name not in arrays:
                values[field.name] = None
                continue
            value = torch.from_numpy(np.array(arrays[field.name][packed_index], copy=True))
            values[field.name] = value.to(getattr(torch, str(arrays[f"__dtype__{field.name}"].item())))
        return PolicyBatch(**values)
