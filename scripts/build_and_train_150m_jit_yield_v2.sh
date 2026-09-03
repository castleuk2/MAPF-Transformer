#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${RAW_ROOT:-$ROOT/pogema-mapf-transformer/data/mapf_lns2_150m}"
PACKED_ROOT="${PACKED_ROOT:-$ROOT/pogema-mapf-transformer/data/mapf_lns2_150m_packed_cpp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/pogema-mapf-transformer/data/mapf_lns2_150m_jit_yield_v2_packed}"
PROJECT="$ROOT/MAPF_Preference_Coordination_Transformer_CPP_Project"

"$PYTHON_BIN" "$ROOT/scripts/build_150m_jit_yield_v2.py" \
  --raw-manifest "$RAW_ROOT/train_manifest.jsonl" \
  --packed-manifest "$PACKED_ROOT/train/manifest.jsonl" \
  --output "$OUTPUT_ROOT/train" --samples 200000 --seed 20260903

"$PYTHON_BIN" "$ROOT/scripts/build_150m_jit_yield_v2.py" \
  --raw-manifest "$RAW_ROOT/val_manifest.jsonl" \
  --packed-manifest "$PACKED_ROOT/val/manifest.jsonl" \
  --output "$OUTPUT_ROOT/val" --samples 20000 --seed 20260904

cd "$PROJECT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_ddp.py \
  --config configs/mapf_lns2_150m_jit_yield_v2_200k_6epoch.yaml \
  --output-dir runs/mpct_lns2_150m_jit_yield_v2_200k_6epoch

