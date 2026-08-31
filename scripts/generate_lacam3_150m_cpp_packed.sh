#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
WORKERS=${WORKERS:-24}
LNS_DATA=${LNS_DATA:-"$ROOT/pogema-mapf-transformer/data/mapf_lns2_150m"}
LACAM_DATA=${LACAM_DATA:-"$ROOT/pogema-mapf-transformer/data/mapf_lacam3_150m"}
PACKED_DATA=${PACKED_DATA:-"$ROOT/pogema-mapf-transformer/data/mapf_lacam3_150m_packed_cpp"}
LACAM_LIB=${LACAM_LIB:-"$ROOT/external/lacam-build/liblacam.so"}
CONFIG=${CONFIG:-"$ROOT/MAPF_Preference_Coordination_Transformer_CPP_Project/configs/mapf_lns2_150m_npz_1epoch.yaml"}

test -s "$LACAM_LIB"

for split in train val; do
  "$PYTHON" "$ROOT/pogema-mapf-transformer/generate_lacam3_from_manifest.py" \
    --manifest "$LNS_DATA/${split}_manifest.jsonl" \
    --source-root "$LNS_DATA" \
    --output-root "$LACAM_DATA" \
    --output-manifest "$LACAM_DATA/${split}_manifest.jsonl" \
    --lib "$LACAM_LIB" \
    --timeout 10 \
    --max-steps 128 \
    --workers "$WORKERS"

  (
    cd "$ROOT/MAPF_Preference_Coordination_Transformer_CPP_Project"
    PYTHONPATH=src "$PYTHON" pack_policy_dataset_cpp.py \
      --config "$CONFIG" \
      --manifest "$LACAM_DATA/${split}_manifest.jsonl" \
      --output-dir "$PACKED_DATA/$split" \
      --workers "$WORKERS"
  )
done

echo "LaCAM3 trajectories: $LACAM_DATA"
echo "C++ packed PolicyBatch: $PACKED_DATA"
