# POGEMA–MAPF Transformer Integration

A separate, non-invasive extension project for running the proposed MAPF
Transformer with POGEMA. It does not vendor or patch POGEMA source files;
instead it adds an environment adapter, execution-feedback records, expert
planning, dataset generation and evaluation. This keeps the integration easier
to update when the upstream simulator changes.

## What is adapted

- Forces/validates POGEMA's `observation_type="MAPF"` global fields:
  `global_obstacles`, `global_xy`, `global_target_xy`; current POGEMA supplies one global coordinate per agent dictionary and the adapter stacks them into `[N,2]`.
- Preserves POGEMA action ordering: `WAIT, UP, DOWN, LEFT, RIGHT`.
- Uses `collision_system="soft"` by default so vertex/edge conflicts become explicit failed movements rather than hidden policy inputs.
- Computes **actual displacement** after each environment step and distinguishes
  successful movement, failed movement and intentional wait.
- Records raw expert trajectories in the policy project's `.npz` format.
- Precomputes online-causal stable neighbor slots without using future frames.
- Provides a POGEMA evaluation loop around
  `MAPFTransformerInference.act()/reset_states()`.

## Installation

Install the sibling policy package first, then this package with the POGEMA
extra:

```bash
python -m pip install -e ../mapf-transformer-policy
python -m pip install -e '.[pogema]'
```

The extra installs the current POGEMA GitHub package. Pin a commit in your own
lock file for reproducible experiments.

## MAPF-LNS2 expert

The default expert is the official MAPF-LNS2 command-line solver. Its source is
not vendored because it has its own USC Research License. Install its build
dependencies and use the setup helper:

```bash
sudo apt install libboost-all-dev libeigen3-dev
./tools/setup_mapf_lns2.sh
```

The bridge writes each POGEMA instance to MovingAI `.map`/`.scen` files, runs
MAPF-LNS2, parses `--outputPaths`, converts paths to POGEMA actions, and validates
all vertex/edge transitions before saving an episode.

## Generate train/validation datasets

The grid configuration follows MAPF-GPT's dataset YAML organization for
separate train/validation map catalogs, scenario seeds, agent-count grids,
algorithm settings and tabular result views. Raw `.npz` episodes and JSONL
manifests remain the source of truth.

```bash
python generate_grid_dataset.py --config configs/dataset_grid_mapf_lns2.yaml
```

Scenario-level multiprocessing is controlled in YAML:

```yaml
generation:
  num_processes: 24
```

Override it for one run with `--workers`, for example:

```bash
python generate_grid_dataset.py --config configs/dataset_grid_mapf_lns2.yaml \
  --max-scenarios-per-split 8 --workers 4
```

Each worker runs one independent POGEMA + MAPF-LNS2 scenario. Only the parent
process writes manifests and result tables, avoiding concurrent metadata writes.

The configuration uses MAPF-GPT's four source map catalogs to create only the
two datasets needed during model training: `train` and `val`. The upstream
catalogs named `eval` are treated as training-time validation maps:

```text
maze/train:    9,998 maps × seeds 0..99 × agents [16,24,32]
maze/val:        200 maps × seeds 0..9  × agents [8,16,24,32]
random/train: 10,000 maps × seeds 0..99 × agents [16,24,32]
random/val:      128 maps × seeds 0..9  × agents [8,16,24,32]
```

Outputs include one manifest under each `maze|random` / `train|val` directory,
plus combined `train_manifest.jsonl`, `val_manifest.jsonl`, `results.csv`,
`results.json`, `dataset_summary.json`, and `resolved_grid_config.yaml` under
`data/mapf_lns2`. Agent-specific final `arrival_steps` are stored per episode;
`num_samples = sum(arrival_steps) = SoC`, so makespan padding WAIT actions after
final goal arrival are excluded. Counts are summarized by train/val,
maze/random, and split. For a
quick integration run without expanding all ~6 million scenarios, add
`--max-scenarios-per-split 1`.

### Dataset YAML parameter reference

| Parameter | Required | Meaning |
|---|---|---|
| `output_root` | yes | Root directory for episodes, manifests, and reports |
| `generation.num_processes` | yes | Number of independent scenario workers; default project value is 24 |
| `max_episode_steps` | yes | Reject longer expert plans and cap POGEMA rollout length |
| `obs_radius` | yes | Must remain 7 for the policy's 15×15 local map |
| `on_target` | yes | `nothing` keeps static MAPF goals/agents unchanged |
| `collision_system` | yes | `soft` exposes conflicts as failed movements |
| `precompute_tracking` | yes | Stores stable neighbor slots in NPZ episodes |
| `tracking_grace_steps` | yes when tracking | Frames retained before an absent neighbor slot is reassigned |
| `compress` | optional | Smaller NPZ files at additional CPU cost |
| `maps_file` | yes | One of the official MAPF-GPT map catalogs |
| `map_indices` | optional | Stop-exclusive catalog-position range, e.g. `{start: 0, stop: 2500}` for PC sharding |
| `map_ids` | optional | Stop-exclusive range matched against the trailing number in each map name |
| `seeds` | yes | POGEMA start/goal scenario seeds for each fixed map |
| `num_agents` | yes | Agent-count grid for each map and seed |
| `binary` | yes | Official MAPF-LNS2 executable |
| `cutoff_time` | yes | Per-scenario MAPF-LNS2 budget; project value is 10 seconds |
| `init_algo`, `replan_algo` | yes | MAPF-LNS2 initial and repair solvers |
| `destroy_strategy` | yes | LNS neighborhood selection heuristic |
| `neighbor_size` | yes | Agents included in one repair neighborhood |
| `max_iterations` | yes | Must be high enough that `cutoff_time`, rather than the iteration count, stops LNS repair |
| `screen` | optional | MAPF-LNS2 console verbosity |
| `results_views` | optional | Console-only group summaries; does not affect generated data |

`map_size` and `density` are intentionally absent. They only control random map
generation, while this pipeline supplies fixed MAPF-GPT map strings. POGEMA
uses each catalog map's actual dimensions and obstacles.

Generated episode fields (using POGEMA's global coordinate frame, including its artificial observation border when present):

- static obstacle map;
- global positions for every frame;
- static goals;
- expert commands;
- optional stable neighbor IDs, validity and track-reset arrays;
- planner/seed/action-order metadata.

Initial history needs no special episode type. The policy data loader pads the
missing leading frames and applies frame-validity masks.

## Train

The companion `train.py` forwards to the policy package:

```bash
python train.py \
  --config ../mapf-transformer-policy/configs/mapf_transformer_base.yaml \
  --train-manifest data/mapf_lns2/train_manifest.jsonl \
  --val-manifest data/mapf_lns2/val_manifest.jsonl
```

Post-training rollout evaluation is outside this dataset-generation pipeline.

### MAPF-GPT-6M benchmark comparison

`benchmark_compare.py` evaluates the current policy and MAPF-GPT-6M on exact
copies of the official `eval_configs`. Both policies receive the same map,
agent count, seed, collision system and episode limit. Results use POGEMA's
definitions: CSR (all agents finish), ISR (fraction finishing), final-stable
arrival SoC, makespan, and policy runtime excluding `env.step` time.

Install the optional official MAPF-GPT inference dependencies once:

```bash
../MAPF/bin/python -m pip install -r requirements-benchmark.txt
```

Run an in-distribution comparison on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=../mapf-transformer-policy/src:src \
../MAPF/bin/python benchmark_compare.py \
  --suites random mazes \
  --models mapf_transformer mapf_gpt_6m \
  --device cuda:0 \
  --output-dir results/mapf_gpt_comparison
```

Add `warehouse movingai puzzles` to `--suites` for all 3,296 official
scenarios per model. Episode rows are flushed to `episodes.csv`; rerunning the
same command resumes by skipping completed model/scenario keys. Aggregates by
model, suite, and agent count are written to `summary.csv` and `summary.json`.

Compare the trained policy with the 10-second MAPF-LNS2 expert on the official
Random and Maze suites:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=../mapf-transformer-policy/src:src \
../MAPF/bin/python benchmark_compare.py \
  --suites random mazes \
  --models mapf_transformer mapf_lns2 \
  --current-checkpoint ../mapf-transformer-policy/runs/mapf_lns2_goal_wait20/last.pt \
  --device cuda:0 \
  --mapf-lns2-binary external/MAPF-LNS2/lns \
  --mapf-lns2-cutoff 10 \
  --mapf-lns2-max-iterations 1000000000 \
  --mapf-lns2-workers 24 \
  --output-dir results/mapf_transformer_vs_lns2
```

MAPF-LNS2 receives the full map, starts, and goals and plans once per scenario.
Its reported runtime is solver planning time. The learned policy receives local
observations at every step; its runtime is the sum of policy inference calls.
Both action sequences are executed in the same POGEMA environment and use the
same CSR, ISR, final-stable SoC, and makespan implementation.
`--mapf-lns2-workers 24` launches 24 independent solver processes; only the
parent process appends completed episode rows to the shared CSV.

### Visualization

POGEMA can record an episode as an animated SVG. Save one SVG per evaluation
episode with:

```bash
python inference.py --config configs/evaluation.yaml --checkpoint PATH/last.pt \
  --episodes 1 --save-svg-dir renders
```

For terminal visualization without saving a file, use ANSI rendering:

```bash
python inference.py --config configs/evaluation.yaml --checkpoint PATH/last.pt \
  --episodes 1 --render-mode ansi
```

`save_svg_dir` and `render_mode` can also be set in `configs/evaluation.yaml`.

## Core modules

- `compat.py`: lazy POGEMA import and API normalization.
- `env_adapter.py`: transition/execution feedback.
- `expert.py`: MAPF-LNS2 bridge and path validation utilities.
- `episode_io.py`: stable tracking precomputation and episode assembly.
- `dataset_generator.py`: POGEMA scenario → expert rollout → `.npz` dataset.
- `grid_dataset_generator.py`: split map/seed/agent grids and result views.
- `policy_adapter.py`: POGEMA-facing policy facade.
- `evaluation.py`: simulation evaluation CLI.
