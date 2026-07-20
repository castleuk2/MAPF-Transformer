# MAPF Transformer Workspace

This workspace contains two separate Python projects:

1. **`mapf-transformer-policy`** — the hierarchical spatial/temporal policy,
   raw-trajectory dataset loader, training CLI and MAPF-GPT-like inference
   interface.
2. **`pogema-mapf-transformer`** — a non-invasive POGEMA extension containing
   environment feedback adaptation, expert planning, trajectory recording,
   dataset generation and simulation evaluation.

## Recommended installation order

```bash
cd mapf-transformer-policy
python -m pip install -e .

cd ../pogema-mapf-transformer
python -m pip install -e '.[pogema]'
```

## End-to-end workflow

```bash
# 1. Generate MAPF-LNS2 expert trajectories in POGEMA
cd pogema-mapf-transformer
python generate_grid_dataset.py --config configs/dataset_grid_mapf_lns2.yaml

# 2. Train the policy
python train.py \
  --config ../mapf-transformer-policy/configs/mapf_transformer_base.yaml \
  --train-manifest data/mapf_lns2/train_manifest.jsonl \
  --val-manifest data/mapf_lns2/val_manifest.jsonl

# 3. Evaluate the checkpoint in POGEMA
python inference.py \
  --config configs/evaluation.yaml \
  --checkpoint ../mapf-transformer-policy/runs/mapf_transformer_base/best.pt
```

MAPF-LNS2 is the default expert and is integrated through its official external
binary. The included prioritized planner remains available only as a lightweight
smoke-test fallback.

## Validation performed in this workspace

- Main policy unit/integration tests.
- Spatial memory incremental-update equivalence against full 15×15 cropping.
- Multi-hot shortest-path action mask and 18-bit payload round-trip.
- Initial-history PAD behavior.
- Model forward/backward pass.
- Stateful `act`/`reset_states` inference wrapper.
- Collision-free prioritized planner and stable tracking tests.
- Synthetic dataset → training steps → checkpoint → offline inference smoke run.

Actual POGEMA execution requires the optional upstream simulator dependency and
is intentionally isolated behind lazy imports.
