#!/usr/bin/env python3
"""Verify exact tensor parity between Python- and C++-generated packed data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def records(path: Path) -> list[tuple[Path, dict]]:
    path = path.resolve()
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        episode = Path(record["path"])
        if not episode.is_absolute():
            episode = path.parent / episode
        output.append((episode.resolve(), record))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-manifest", type=Path, required=True)
    parser.add_argument("--cpp-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=0, help="0 checks every episode")
    args = parser.parse_args()

    python_rows = records(args.python_manifest)
    cpp_rows = records(args.cpp_manifest)
    if len(python_rows) != len(cpp_rows):
        raise AssertionError(f"episode count differs: {len(python_rows)} != {len(cpp_rows)}")
    limit = len(python_rows) if args.episodes == 0 else min(args.episodes, len(python_rows))
    arrays_checked = samples = 0
    for index, ((python_path, python_record), (cpp_path, cpp_record)) in enumerate(
        zip(python_rows[:limit], cpp_rows[:limit])
    ):
        if python_record["samples"] != cpp_record["samples"]:
            raise AssertionError(f"sample count differs at episode {index}")
        if python_record["sample_counts"] != cpp_record["sample_counts"]:
            raise AssertionError(f"per-agent sample counts differ at episode {index}")
        samples += int(python_record["samples"])
        with np.load(python_path, allow_pickle=False) as python_data, np.load(
            cpp_path, allow_pickle=False
        ) as cpp_data:
            if set(python_data.files) != set(cpp_data.files):
                raise AssertionError(f"fields differ at episode {index}")
            for field in python_data.files:
                if not np.array_equal(python_data[field], cpp_data[field]):
                    raise AssertionError(f"tensor differs: episode={index}, field={field}")
                arrays_checked += 1
    print(
        f"exact_tensor_parity=OK episodes={limit} samples={samples} "
        f"arrays={arrays_checked}"
    )


if __name__ == "__main__":
    main()
