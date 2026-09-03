# MPCT 150M JIT-Yield v2 experiment

Branch: `mpct-lns2-150m-jit-yield-v2`

This experiment scans the existing 150M MAPF-LNS2 trajectories and selects a
new 200K Train / 20K Validation view whose Goal-state and yield-timing marginals
match the measured LaCAM3 distribution.  It does not modify features or labels:
the manifests reference the original C++ precomputed Packed PolicyBatch files.

The purpose is to compare against the earlier 200K experiment at exactly the
same training exposure while replacing repeated rare samples with diverse
examples drawn from the much larger corpus.  The builder stops with an error if
the 150M source still lacks enough unique samples.  Repetition is enabled only
when `--allow-repeat` is explicitly supplied.

## Expected input

```text
pogema-mapf-transformer/data/mapf_lns2_150m/
├── train_manifest.jsonl
└── val_manifest.jsonl

pogema-mapf-transformer/data/mapf_lns2_150m_packed_cpp/
├── train/manifest.jsonl
└── val/manifest.jsonl
```

The raw trajectories are needed to identify Goal departure, return, and the
time at which another Agent uses the vacated Goal.  Packed files supply the
unchanged model inputs and labels used during training.

## Run the complete pipeline

```bash
git clone --branch mpct-lns2-150m-jit-yield-v2 --single-branch \
  https://github.com/castleuk2/MAPF-Transformer.git
cd MAPF-Transformer

python -m pip install -e mapf-structured-map-transformer --no-deps
python -m pip install -e MAPF_Preference_Coordination_Transformer_CPP_Project \
  --no-build-isolation --no-deps

PYTHON_BIN="$(command -v python)" CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/build_and_train_150m_jit_yield_v2.sh
```

If the local directories use different names, set `RAW_ROOT`, `PACKED_ROOT`,
and `OUTPUT_ROOT` before the command.  Dataset construction is deterministic
with the recorded seeds and uses bounded reservoirs, so it does not hold all
125M sample indices in RAM.

Training matches the earlier small experiment: frozen Structured Map Encoder,
global batch 256, two-GPU DDP, six epochs, identical optimizer and loss weights.
Outputs are written to:

```text
MAPF_Preference_Coordination_Transformer_CPP_Project/
└── runs/mpct_lns2_150m_jit_yield_v2_200k_6epoch/
```
