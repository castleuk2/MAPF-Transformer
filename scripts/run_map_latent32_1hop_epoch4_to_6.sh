#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
output=mapf-transformer-policy/runs/ablation_map_latent32_one_hop_ctg_6epoch
mkdir -p "$output"
cp --update=none mapf-transformer-policy/checkpoints_3epoch/map_latent32_1hop_best_epoch3.pt \
  "$output/best.pt"

python_bin=${PYTHON:-python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} "$python_bin" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  mapf-transformer-policy/train.py \
  --config mapf-transformer-policy/configs/ablation_map_latent32_one_hop_ctg_6epoch.yaml \
  --resume mapf-transformer-policy/checkpoints_3epoch/map_latent32_1hop_last_epoch3.pt \
  2>&1 | tee "$output/console.log"
