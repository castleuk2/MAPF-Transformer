# Three-model six-epoch continuation

The comparison fixes the three learned policies to:

1. MAPF-GPT-6M
2. Map Latent 32 + one-hop CTG
3. MPCT with the C++ feature generator

Each run keeps its epoch-3 `last` and `best` checkpoints unchanged. Training for
epochs 4-6 always resumes from epoch-3 `last`; epoch-3 `best` is retained as the
initial six-epoch `best`, and is replaced only when validation improves.

## Why checkpoints are transferred separately

Git contains code and configurations only. The Map Latent checkpoint is over
GitHub's 100 MiB per-file limit, and datasets/checkpoints are machine artifacts.
Transfer the `.pt` files with `scp`/`rsync`, then verify SHA-256 checksums.

On the source machine:

```bash
cd /home/vialab/Seonguk/MAPF-Transformer-6epoch
bash scripts/prepare_3epoch_checkpoints.sh \
  /home/vialab/Seonguk/MAPF-Transformer-workspace
```

Transfer to the other PC (replace the host and path):

```bash
rsync -avP mapf-gpt-mapf-lns2/checkpoints_3epoch/ USER@HOST:MAPF-Transformer/mapf-gpt-mapf-lns2/checkpoints_3epoch/
rsync -avP mapf-transformer-policy/checkpoints_3epoch/ USER@HOST:MAPF-Transformer/mapf-transformer-policy/checkpoints_3epoch/
rsync -avP checkpoints_3epoch.sha256 USER@HOST:MAPF-Transformer/
```

On the destination PC:

```bash
cd MAPF-Transformer
sha256sum -c checkpoints_3epoch.sha256
```

The destination must already contain the same Arrow/NPZ datasets at the paths
specified by the configs. Checkpoints alone do not contain training data.

## Run on two GPUs

MAPF-GPT-6M:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_mapf_gpt_6m_epoch4_to_6.sh
```

Map Latent 32 + one-hop CTG:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_map_latent32_1hop_epoch4_to_6.sh
```

MPCT C++ after the current epoch-3 run finishes:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_mpct_cpp_epoch4_to_6.sh /absolute/path/to/epoch3/last.pt
```

Do not resume from `best.pt`: its epoch can precede epoch 3. `last.pt` defines a
true three-plus-three continuation; `best.pt` is for model selection/evaluation.
