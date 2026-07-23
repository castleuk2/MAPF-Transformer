#!/usr/bin/env bash
set -euo pipefail

GRID_CONFIG="mapf-transformer-policy/configs/agent_latent_grid_raw_1h.yaml"

for MAP_LATENTS in 32 48 64; do
  for AGENT_LATENTS in 32 48 64; do
    RUN_NAME="raw_map${MAP_LATENTS}_agent${AGENT_LATENTS}"
    RUN_DIR="mapf-transformer-policy/runs/${RUN_NAME}"
    mkdir -p "${RUN_DIR}"

    PYTHONPATH=mapf-transformer-policy/src \
      python mapf-transformer-policy/prepare_agent_latent_grid.py \
        --grid-config "${GRID_CONFIG}" \
        --map-latents "${MAP_LATENTS}" \
        --agent-latents "${AGENT_LATENTS}"

    CUDA_VISIBLE_DEVICES=0,1 \
    PYTHONPATH=mapf-transformer-policy/src \
      torchrun --standalone --nproc_per_node=2 \
        mapf-transformer-policy/train.py \
        --config "${RUN_DIR}/launch_config.yaml" \
        2>&1 | tee "${RUN_DIR}/console.log"
  done
done
