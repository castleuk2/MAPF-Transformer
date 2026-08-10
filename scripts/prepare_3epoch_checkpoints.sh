#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${1:?usage: $0 /path/to/source/MAPF-Transformer-workspace}

gpt_dir="$repo_root/mapf-gpt-mapf-lns2/checkpoints_3epoch"
map_dir="$repo_root/mapf-transformer-policy/checkpoints_3epoch"
mkdir -p "$gpt_dir" "$map_dir"

cp -n "$source_root/mapf-gpt-mapf-lns2/runs/mapf_gpt_6m_mapf_lns2/last.pt" \
  "$gpt_dir/mapf_gpt_6m_last_epoch3.pt"
cp -n "$source_root/mapf-gpt-mapf-lns2/runs/mapf_gpt_6m_mapf_lns2/ckpt.pt" \
  "$gpt_dir/mapf_gpt_6m_best_epoch3.pt"
cp -n "$source_root/mapf-transformer-policy/runs/ablation_map_latent32_one_hop_ctg/last.pt" \
  "$map_dir/map_latent32_1hop_last_epoch3.pt"
cp -n "$source_root/mapf-transformer-policy/runs/ablation_map_latent32_one_hop_ctg/best.pt" \
  "$map_dir/map_latent32_1hop_best_epoch3.pt"

cd "$repo_root"
sha256sum mapf-gpt-mapf-lns2/checkpoints_3epoch/*.pt \
  mapf-transformer-policy/checkpoints_3epoch/*.pt > checkpoints_3epoch.sha256
echo "Prepared checkpoints and checksums under $repo_root"
