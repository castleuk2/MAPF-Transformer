#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../mapf-gpt-mapf-lns2" && pwd)
cd "$project_root"
mkdir -p runs/mapf_gpt_6m_mapf_lns2_6epoch
cp --update=none checkpoints_3epoch/mapf_gpt_6m_best_epoch3.pt \
  runs/mapf_gpt_6m_mapf_lns2_6epoch/best.pt

python_bin=${PYTHON:-python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} "$python_bin" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  train.py experiment_setup/config-6M-mapf-lns2-6epoch.py \
  2>&1 | tee runs/mapf_gpt_6m_mapf_lns2_6epoch/console.log
