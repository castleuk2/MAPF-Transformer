# Packed frame cache

The raw expert trajectories remain the source of truth. The packed cache
precomputes each `(episode, frame, ego)` feature exactly once and does not
materialize repeated 15-frame training samples.

Per frame and ego it stores:

- the binary 15x15 local map as 225 bits (29 bytes);
- x(4), y(4), shortest-path mask(4), and distance(6) as one 18-bit agent payload;
- validity and track-reset as 16-bit slot masks;
- transition categories as `uint8`;
- optional one-hop CTG categories only when enabled by the conversion config.

The DataLoader selects the required history in compact form. After the batch is
transferred, the GPU adapter losslessly restores the original tensors. Model
parameters and checkpoints are unchanged.

## Convert the one-hour baseline

Run from the workspace root:

```bash
PYTHONPATH=mapf-transformer-policy/src MAPF/bin/python \
  mapf-transformer-policy/pack_dataset.py \
  --manifest pogema-mapf-transformer/data/mapf_lns2_1h/train_manifest.jsonl \
  --output-dir pogema-mapf-transformer/data/mapf_lns2_1h_packed/train \
  --config mapf-transformer-policy/configs/ablation_baseline_latent16.yaml \
  --workers 24

PYTHONPATH=mapf-transformer-policy/src MAPF/bin/python \
  mapf-transformer-policy/pack_dataset.py \
  --manifest pogema-mapf-transformer/data/mapf_lns2_1h/val_manifest.jsonl \
  --output-dir pogema-mapf-transformer/data/mapf_lns2_1h_packed/val \
  --config mapf-transformer-policy/configs/ablation_baseline_latent16.yaml \
  --workers 24
```

Completed episode files are skipped, so an interrupted conversion can be run
again. Do not use `--overwrite` when resuming. The default cache is deliberately
uncompressed to minimize training-time CPU decompression; add `--compress` only
when disk size matters more than loader throughput.

## Train

```bash
mkdir -p mapf-transformer-policy/runs/packed_baseline_latent16
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=mapf-transformer-policy/src \
torchrun --standalone --nproc_per_node=2 \
  mapf-transformer-policy/train.py \
  --config mapf-transformer-policy/configs/packed_baseline_latent16.yaml
```

Packed caches are feature-configuration specific. A one-hop CTG model requires
a cache generated with a config where `model.one_hop_ctg: true`.
