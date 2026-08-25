from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mapf_pct.config import load_config
from mapf_pct.model import PreferenceCoordinationTransformer


EXPECTED_MAP_SHA256 = "4d3ebbb73f90f2bcf496fca328f5d20192194fca72e63c71a8307c0e335632b2"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def manifest_summary(path: Path) -> tuple[int, int]:
    records = 0
    missing = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        episode = Path(record["path"])
        if not episode.is_absolute():
            episode = path.parent / episode
        records += 1
        missing += int(not episode.exists())
    return records, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify portable MPCT training assets")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_dataset_frozen_map.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    checkpoint = resolve(config.model.map_checkpoint or "")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"map checkpoint not found: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_MAP_SHA256:
        raise RuntimeError(f"map checkpoint SHA256 mismatch: {digest}")

    for name, value in (("train", config.data.train_manifest), ("val", config.data.val_manifest)):
        if value is None:
            raise ValueError(f"{name}_manifest is missing")
        manifest = resolve(value)
        if not manifest.is_file():
            raise FileNotFoundError(f"{name} manifest not found: {manifest}")
        records, missing = manifest_summary(manifest)
        if missing:
            raise FileNotFoundError(f"{name} manifest has {missing}/{records} missing NPZ files")
        print(f"{name}: records={records}, missing=0, manifest={manifest}")

    model = PreferenceCoordinationTransformer(config.model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(f"map checkpoint: OK ({digest})")
    print(f"model load: OK, trainable={trainable:,}, frozen={frozen:,}")
    print("portable setup: OK")


if __name__ == "__main__":
    main()
