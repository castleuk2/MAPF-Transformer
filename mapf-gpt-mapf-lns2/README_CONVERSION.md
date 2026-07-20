# MAPF-LNS2 trajectory to MAPF-GPT Arrow converter

This directory contains only the source required to convert the shared
MAPF-LNS2 NPZ trajectories into the official MAPF-GPT 256-token Arrow format.
Generated Arrow shards, checkpoints, and compiled C++ extensions are excluded
from Git.

## Install

From the workspace root, with the `MAPF` virtual environment activated:

```bash
python -m pip install -r mapf-gpt-mapf-lns2/requirements-mapf-lns2.txt
```

The tokenizer C++ extensions are built automatically by `cppimport` on the
first conversion run. A C++ compiler and Python development headers are
therefore required.

## Convert the 150M expansion

```bash
python mapf-gpt-mapf-lns2/convert_npz_to_arrow.py \
  --manifest pogema-mapf-transformer/data/mapf_lns2_150m_expansion/train_manifest.jsonl \
  --output-dir mapf-gpt-mapf-lns2/data/mapf_lns2_150m_expansion/train \
  --workers 24 \
  --shard-size 65536 \
  --goal-wait-keep-ratio 0.2

python mapf-gpt-mapf-lns2/convert_npz_to_arrow.py \
  --manifest pogema-mapf-transformer/data/mapf_lns2_150m_expansion/val_manifest.jsonl \
  --output-dir mapf-gpt-mapf-lns2/data/mapf_lns2_150m_expansion/validation \
  --workers 24 \
  --shard-size 65536 \
  --goal-wait-keep-ratio 0.2
```

Every successful trajectory is converted using the same `(episode, ego,
timestep)` selection rule as MAPF Transformer: all pre-arrival SoC targets and
an evenly distributed 20% sample of the final-goal WAIT suffix.

## Verify

Each output directory receives a `conversion_summary.json`. Check its
`episodes`, `samples`, `soc_samples`, `goal_wait_samples`, and `shards` fields.

```bash
python -m pytest mapf-gpt-mapf-lns2/tests/test_npz_conversion.py -q
```
