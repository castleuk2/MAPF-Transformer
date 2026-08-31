# MPCT training with the 150M MAPF-LNS2 dataset

Branch: `mpct-mapf-lns2-150m-packed`

This branch keeps the frozen structured Map Encoder and existing MPCT model/losses.
It adds lossless packed PolicyBatch conversion and two-GPU DDP training for the
150M MAPF-LNS2 dataset.

## 1. Clone and install

```bash
git clone --branch mpct-mapf-lns2-150m-packed --single-branch \
  https://github.com/castleuk2/MAPF-Transformer.git
cd MAPF-Transformer/MAPF_Preference_Coordination_Transformer_CPP_Project

python -m pip install -e . --no-build-isolation
python build_cpp_extension.py
python verify_setup.py
pytest -q
```

## 2. Expected dataset paths

Place or symlink the dataset as follows:

```text
MAPF-Transformer/pogema-mapf-transformer/data/mapf_lns2_150m/
├── train_manifest.jsonl
├── val_manifest.jsonl
└── ... episode .npz files referenced by the manifests
```

Manifest records must contain `path`, `arrival_steps`, `time_steps`, `num_agents`,
and `map_family`. Relative episode paths are resolved from the manifest directory.

```bash
python validate_150m_manifests.py \
  --train-manifest ../pogema-mapf-transformer/data/mapf_lns2_150m/train_manifest.jsonl \
  --val-manifest ../pogema-mapf-transformer/data/mapf_lns2_150m/val_manifest.jsonl
```

This checks all referenced files and calculates the policy-sample count after the
same 20% post-goal WAIT retention used by prior MPCT training.

## 3A. Train directly from source NPZ

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_ddp.py \
  --config configs/mapf_lns2_150m_npz_1epoch.yaml \
  --output-dir runs/mpct_mapf_lns2_150m_npz_1epoch
```

## 3B. Convert to lossless packed data

Packed conversion changes storage/loading only. It preserves sample indices,
every PolicyBatch tensor, dtype, label, and the 20% Goal-WAIT rule.

```bash
python pack_policy_dataset.py \
  --config configs/mapf_lns2_150m_npz_1epoch.yaml \
  --manifest ../pogema-mapf-transformer/data/mapf_lns2_150m/train_manifest.jsonl \
  --output-dir ../pogema-mapf-transformer/data/mapf_lns2_150m_packed/train \
  --workers 24

python pack_policy_dataset.py \
  --config configs/mapf_lns2_150m_npz_1epoch.yaml \
  --manifest ../pogema-mapf-transformer/data/mapf_lns2_150m/val_manifest.jsonl \
  --output-dir ../pogema-mapf-transformer/data/mapf_lns2_150m_packed/val \
  --workers 24
```

The converter is resumable: existing episode packs are reused unless `--overwrite`
is supplied. Each output directory receives `manifest.jsonl` and `metadata.json`.

## 3C. Precompute with the C++ Feature Generator

Use the C++ entry point for the standard MPCT packing path. It generates all Ego
features once per frame, attaches the same expert labels, and writes the same
lossless `PolicyBatch` format consumed by `PackedPolicyDataset`.

```bash
python build_cpp_extension.py

python pack_policy_dataset_cpp.py \
  --config configs/mapf_lns2_150m_npz_1epoch.yaml \
  --manifest ../pogema-mapf-transformer/data/mapf_lns2_150m/train_manifest.jsonl \
  --output-dir ../pogema-mapf-transformer/data/mapf_lns2_150m_packed_cpp/train \
  --workers 24

python pack_policy_dataset_cpp.py \
  --config configs/mapf_lns2_150m_npz_1epoch.yaml \
  --manifest ../pogema-mapf-transformer/data/mapf_lns2_150m/val_manifest.jsonl \
  --output-dir ../pogema-mapf-transformer/data/mapf_lns2_150m_packed_cpp/val \
  --workers 24
```

Generated manifests mark every record with `feature_generator: "cpp"` and
`exact_roundtrip_verified: true`. Packing changes feature-computation time and
storage only; the training loader and model are unchanged.

For a direct Python/C++ audit, generate both outputs from the same source manifest
and compare them. `--episodes 0` checks the complete dataset.

```bash
python verify_cpp_python_packed.py \
  --python-manifest /path/to/python-packed/manifest.jsonl \
  --cpp-manifest /path/to/cpp-packed/manifest.jsonl \
  --episodes 100
```

## 4. Verify source NPZ and packed equality

```bash
python verify_packed_policy.py \
  --config configs/mapf_lns2_150m_npz_1epoch.yaml \
  --source-manifest ../pogema-mapf-transformer/data/mapf_lns2_150m/train_manifest.jsonl \
  --packed-manifest ../pogema-mapf-transformer/data/mapf_lns2_150m_packed/train/manifest.jsonl \
  --samples 10000
```

Use `--samples 0` for exhaustive equality verification. This can take a long time
for 150M samples.

## 5. Train from packed data on two GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_ddp.py \
  --config configs/mapf_lns2_150m_packed_1epoch.yaml \
  --output-dir runs/mpct_mapf_lns2_150m_packed_1epoch
```

Outputs are `best.pt`, `last.pt`, `metrics.jsonl`, and `resolved_config.yaml` under
the selected run directory.

## Epoch and batch interpretation

The default is one epoch because one full pass already consumes approximately
150M policy samples. Setting `epochs: 6` means approximately 900M sample exposures;
it is not equivalent to the previous small-dataset six-epoch run.

Global batch size is 256: with two GPUs, each process receives 128 samples per
optimizer update. Packed conversion does not change batch size or optimizer-update
count; it removes repeated online feature construction.
