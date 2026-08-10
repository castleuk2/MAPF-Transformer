#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../MAPF_Preference_Coordination_Transformer_CPP_Project" && pwd)
checkpoint=${1:?usage: $0 /path/to/mpct_epoch3/last.pt [/path/to/mpct_epoch3/best.pt]}
epoch3_best=${2:-$(dirname "$checkpoint")/best.pt}
cd "$project_root"
mkdir -p runs/mpct_balanced_val_6epoch
if [[ -f "$epoch3_best" ]]; then
  cp -n "$epoch3_best" runs/mpct_balanced_val_6epoch/best.pt
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun --standalone --nproc_per_node=2 \
  train_ddp.py --config configs/balanced_val_6epoch.yaml \
  --output-dir runs/mpct_balanced_val_6epoch --resume "$checkpoint" \
  2>&1 | tee runs/mpct_balanced_val_6epoch/console.log
