# MAPF-LNS2 training-data generation bundle

This bundle generates MAPF Transformer `train` and `val` NPZ episodes on CPU.
It does not contain training or post-training evaluation code and does not
require PyTorch or CUDA.

## 1. System requirements (Ubuntu 24.04)

```bash
sudo apt update
sudo apt install -y python3-venv git cmake build-essential libboost-all-dev libeigen3-dev
```

## 2. Python environment

Run these commands from the extracted bundle root:

```bash
python3 -m venv MAPF
source MAPF/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r data_generation_requirements.txt
```

## 3. Build the official MAPF-LNS2 solver

```bash
bash pogema-mapf-transformer/tools/setup_mapf_lns2.sh
```

## 4. Choose a map shard

Edit `pogema-mapf-transformer/configs/dataset_grid_mapf_lns2.yaml`.
`map_indices` is a zero-based, stop-exclusive range in catalog order:

```yaml
map_indices: {start: 0, stop: 2500}
```

Use disjoint ranges on different PCs. Alternatively, `map_ids` selects the
trailing number in a map name; for example `{start: 640, stop: 1000}` matches
names ending in 00640 through 00999. Only one of `maps`, `map_indices`, and
`map_ids` may be present in a split.

Give every PC a distinct `output_root`, or keep the default locally and merge
the resulting NPZ files/manifests later. Do not have multiple PCs write to the
same network directory concurrently.

## 5. Generate data

```bash
cd pogema-mapf-transformer
nohup env PYTHONPATH=src ../MAPF/bin/python -u generate_grid_dataset.py \
  --config configs/dataset_grid_mapf_lns2.yaml \
  > dataset_generation.log 2>&1 &
echo $! > dataset_generation.pid
```

Monitor with:

```bash
tail -f dataset_generation.log
find data/mapf_lns2 -name '*.npz' | wc -l
```

After completion, `dataset_summary.json` records generated/failed episode
counts and the actual number of learning samples (`sum(arrival_steps) = SoC`)
for train, val, maze, random, and every split. An agent's action that reaches
its final goal is included; only subsequent makespan-padding WAIT actions are
excluded.

The default configuration uses 24 worker processes, a 10-second MAPF-LNS2
cutoff, and a very high iteration ceiling so repair continues until the time
cutoff after an initial solution is found.

For the balanced one-hour trial (3,668 maze and 3,668 random scenarios), the
MAPF-GPT generation-grid ratio is calculated as maps * seeds * agent-count
values. The original ratio is 457.27:1 and this reduced grid is 457.5:1:

```bash
cd pogema-mapf-transformer
nohup env PYTHONPATH=src ../MAPF/bin/python -u generate_grid_dataset.py \
  --config configs/dataset_grid_mapf_lns2_1h.yaml \
  > dataset_generation_1h.log 2>&1 &
```
