# MAPF Transformer

POGEMA 환경에서 MAPF-LNS2 expert trajectory를 생성하고, 계층형
spatio-temporal Transformer를 imitation learning으로 학습·평가하기 위한
workspace입니다. MAPF-GPT의 map/seed/agent 구성과 공식 evaluation suite를
재사용하되, 모델 입력 구조와 trajectory 저장 방식은 별도로 구현했습니다.

현재 저장소는 다음 실험을 재현할 수 있습니다.

- MAPF-GPT의 Maze/Random map catalog로 train/validation trajectory 생성
- 공식 MAPF-LNS2를 expert solver로 사용(시나리오당 10초, 24 CPU process)
- final-goal 이후 WAIT target의 20%를 복원한 학습
- 2 GPU DDP 학습과 구조화된 loss/accuracy log 기록
- map latent 16/32 및 one-hop cost-to-go ablation
- 동일한 MAPF-GPT evaluation scenario에서 MAPF Transformer, MAPF-GPT-6M,
  MAPF-LNS2의 SR, SoC, makespan, runtime 비교
- raw trajectory를 유지하면서 I/O 비용을 줄이는 lossless packed cache

## 저장소 구조

```text
MAPF-Transformer-workspace/
├── mapf-transformer-policy/       # 모델, Dataset/DataLoader, 학습·추론 코드
├── pogema-mapf-transformer/       # POGEMA adapter, MAPF-LNS2 데이터 생성·평가
├── mapf-gpt-mapf-lns2/            # 동일 expert data용 MAPF-GPT-6M 학습·추론 코드
├── dataset_archives/              # Git LFS로 관리되는 150M trajectory archive
├── DATA_GENERATION_README.md       # 데이터 생성 전용 PC 구성 안내
└── VALIDATION.md                   # 초기 검증 기록
```

세 프로젝트는 독립적인 Python package이며, POGEMA나 MAPF-GPT 원본을 직접
수정하지 않습니다.

## 모델 구조

기본 모델은 약 8.8M parameter의 hierarchical Transformer입니다.

- ego 중심 15x15 local obstacle map
- 225 cell token을 cross-attention으로 16개 learned map latent로 압축
- ego 1개와 최대 15개 neighbor를 안정적인 tracking slot에 배치
- agent payload: local x(4 bit), y(4 bit), shortest-path action mask(4 bit),
  distance bucket(6 bit)
- frame당 conditioned agent token 16개 + transition token 1개
- 최근 15 frame(255 token) + `[ACT]` query 1개 = temporal context 256 token
- temporal Transformer 8 layer, `d_model=256`, 8 attention head
- POGEMA action 순서: `WAIT, UP, DOWN, LEFT, RIGHT`

초기 시점에 존재하지 않는 history frame은 validity mask가 false인 PAD frame으로
처리합니다. Local map이 global map 경계를 넘어갈 때는 바깥 영역을 obstacle로
padding합니다. 모델 입력은 현재 시점 하나의 행동 label을 예측하지만, 그 입력에는
최대 15개 과거 frame의 local map, 주변 agent, transition 정보가 포함됩니다.

자세한 구조는 [policy README](mapf-transformer-policy/README.md)를 참고하십시오.

## 환경 구축

Ubuntu와 Python 3.10 이상을 지원합니다. 재현성을 위해 현재 주 실험 환경과 같은
Python 3.12를 권장합니다. Python 3.13에서는 설치 시점의 최신 PyTorch/CUDA wheel이
선택될 수 있으므로 두 PC의 package version을 반드시 비교하십시오.

```bash
git clone https://github.com/castleuk2/MAPF-Transformer.git
cd MAPF-Transformer

python3 -m venv MAPF
source MAPF/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install -e mapf-transformer-policy
python -m pip install -e 'pogema-mapf-transformer[pogema]'
python -m pip install -r pogema-mapf-transformer/requirements-benchmark.txt
```

GPU 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

### MAPF-LNS2 설치

MAPF-LNS2는 별도 license를 사용하므로 binary를 저장소에 포함하지 않습니다.

```bash
sudo apt install -y cmake build-essential libboost-all-dev libeigen3-dev
bash pogema-mapf-transformer/tools/setup_mapf_lns2.sh
```

기본 config가 참조하는 binary는
`pogema-mapf-transformer/external/MAPF-LNS2/lns`입니다.

## 데이터 생성

### 데이터 단위와 filtering

원본 `.npz` episode에는 obstacle map, 모든 agent의 positions/goals/actions,
stable neighbor tracking 정보, 각 agent의 `arrival_steps`가 저장됩니다.

학습 sample 하나는 `(episode, ego agent, time step)`이며 target은 해당 ego의
expert action 하나입니다. Solver 실패, 128 step 초과, vertex/edge transition 검증
실패 episode는 manifest에 포함하지 않으므로 학습에서도 자동 제외됩니다.

기본 sample 수는 다음과 같습니다.

```text
base samples per episode = sum(arrival_steps) = expert SoC
```

목표에 최종적으로 도착하기 전의 행동(목표에서 다시 나와 양보하는 행동 포함)은
모두 유지합니다. 최종 도착 이후 makespan까지 이어지는 반복 WAIT은 기본적으로
제외하지만, 학습 loader가 `goal_wait_keep_ratio: 0.2`에 따라 그중 약 20%를
결정적으로 복원합니다. 따라서 실제 학습 sample 수는 base SoC보다 큽니다.

### 1시간 실험 데이터

```bash
cd pogema-mapf-transformer
PYTHONPATH=src ../MAPF/bin/python -u generate_grid_dataset.py \
  --config configs/dataset_grid_mapf_lns2_1h.yaml
```

생성 설정은 Maze/Random 각각 train `122 maps x 10 seeds x 3 agent counts`
(`[16,24,32]`)이며, validation은 별도 held-out map catalog를 사용합니다.
최종 저장본은 validation을 확장·교체한 결과이며 다음 성공 episode를 포함합니다.

| Split | 성공 episode | Base SoC sample |
|---|---:|---:|
| Train | 6,988 | 2,819,063 |
| Validation | 739 | 284,074 |
| 합계 | 7,727 | 3,103,137 |

시도한 8,088개 중 361개 실패 episode는 제외되었습니다. Map family별 base sample은
Maze 1,955,070(63.0%), Random 1,148,067(37.0%)입니다. 정확한 수치는
`pogema-mapf-transformer/data/mapf_lns2_1h/dataset_summary.json`에 기록됩니다.

### 약 150M train / 15M validation 확장 데이터

```bash
cd pogema-mapf-transformer
PYTHONPATH=src ../MAPF/bin/python -u generate_grid_dataset.py \
  --config configs/dataset_grid_mapf_lns2_150m_expansion.yaml
```

기존 map/seed를 반복하지 않고 map 다양성만 확장합니다. Train은 family마다 신규
5,400 maps x 10 seeds x agents `[16,24,32]`, validation은 family마다 held-out
2,820 maps x 2 seeds x 같은 agent 수를 사용합니다. 24 process가 서로 독립적인
scenario를 병렬 처리하며, MAPF-LNS2는 높은 iteration ceiling 아래 10초 time limit을
모두 repair에 사용할 수 있습니다.

Git LFS archive를 받은 경우 다음과 같이 복원합니다.

```bash
git lfs install
git lfs pull
cd pogema-mapf-transformer/data
mkdir -p mapf_lns2_150m_expansion
cd mapf_lns2_150m_expansion
tar -xf ../../../dataset_archives/mapf_lns2_150m_maze_train.tar
tar -xf ../../../dataset_archives/mapf_lns2_150m_random_train.tar
tar -xf ../../../dataset_archives/mapf_lns2_150m_maze_val.tar
tar -xf ../../../dataset_archives/mapf_lns2_150m_random_val.tar
tar -xzf ../../../dataset_archives/mapf_lns2_150m_metadata.tar.gz
```

Archive의 checksum은 `dataset_archives/SHA256SUMS`로 확인할 수 있습니다.

## Lossless packed cache

Raw trajectory는 작고 모델 구조 변경에 재사용할 수 있지만, 매 학습 sample마다
15-frame feature를 CPU에서 동적으로 구성하는 비용이 큽니다. Packed cache는
각 `(episode, frame, ego)` feature를 한 번만 전처리하며, 반복된 15-frame sample을
저장하지 않습니다.

- 15x15 binary map: 225 bit -> 29 byte bitmap
- agent x/y/action-mask/distance: 18-bit payload
- valid/reset: 16-bit slot mask
- transition 및 선택적 one-hop CTG: `uint8`
- batch 전송 후 GPU에서 기존 tensor와 losslessly 동일하게 복원

실측 기준 DataLoader sample 표현은 47,199 byte에서 1,614 byte로 약 29.2배
감소했습니다. 이는 입력 I/O와 CPU feature 처리 최적화이며 모델, logits, loss,
checkpoint 형식은 바꾸지 않습니다. 전체 설명과 검증 결과는
[PACKED_DATASET.md](mapf-transformer-policy/PACKED_DATASET.md)에 있습니다.

1시간 데이터 변환:

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

중단 후 같은 명령을 다시 실행하면 완료 episode를 건너뜁니다. Resume할 때
`--overwrite`는 사용하지 마십시오. One-hop CTG cache는 feature가 다르므로
`ablation_one_hop_ctg.yaml`로 별도 생성해야 합니다.

## 학습

### 실험 config

| Config | Map latent | One-hop CTG | 입력 형식 |
|---|---:|---:|---|
| `ablation_baseline_latent16.yaml` | 16 | off | raw trajectory |
| `ablation_map_latent32.yaml` | 32 | off | raw trajectory |
| `ablation_one_hop_ctg.yaml` | 16 | on | raw trajectory |
| `packed_baseline_latent16.yaml` | 16 | off | packed cache |

세 ablation config는 비교 대상 외 모델과 optimizer 설정을 동일하게 유지합니다.
현재 기본 학습은 3 epoch, AdamW, peak LR `3e-4`, cosine decay, effective batch
256, AMP를 사용합니다.

### 2 GPU DDP 실행

먼저 log directory를 생성해야 `tee`가 시작과 동시에 실패하지 않습니다.

```bash
mkdir -p mapf-transformer-policy/runs/ablation_baseline_latent16
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=mapf-transformer-policy/src \
torchrun --standalone --nproc_per_node=2 \
  mapf-transformer-policy/train.py \
  --config mapf-transformer-policy/configs/ablation_baseline_latent16.yaml \
  2>&1 | tee mapf-transformer-policy/runs/ablation_baseline_latent16/console.log
```

`batch_size`는 GPU 하나가 한 micro-step에 처리하는 sample 수입니다.
`gradient_accumulation_steps`는 모든 rank를 합친 global 값이며 world size로
나뉩니다. 따라서 2 GPU에서 `batch_size=16`, accumulation 16이면 effective batch는
256입니다. Packed cache에서 동일한 effective batch를 유지하며 GPU utilization을
높이려면 우선 micro-batch 64/accumulation 4, 여유가 있으면 128/2를 비교하십시오.

1 GPU는 `CUDA_VISIBLE_DEVICES=0`과 `--nproc_per_node=1`로 선택할 수 있습니다.

### 재시작과 출력

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=mapf-transformer-policy/src \
torchrun --standalone --nproc_per_node=2 \
  mapf-transformer-policy/train.py \
  --config mapf-transformer-policy/configs/ablation_baseline_latent16.yaml \
  --resume mapf-transformer-policy/runs/ablation_baseline_latent16/step_00005000.pt
```

Rank 0은 다음 파일을 기록합니다.

- `console.log`: 표준 출력/경고/오류 전체(`tee` 사용 시)
- `metrics.jsonl`: step별 total/action/map loss, LR, throughput과 validation metric
- `resolved_config.yaml`: 실제 적용 설정
- `best.pt`, `step_*.pt`, `last.pt`: checkpoint

Total loss는 다음과 같습니다.

```text
total_loss = action_cross_entropy + 0.05 * map_reconstruction_BCE
```

Action loss는 5개 행동에 대한 cross-entropy입니다. Map reconstruction loss는
유효한 history frame의 225개 cell을 map latent에서 복원한 binary cross-entropy로,
0에 가까울수록 원래 obstacle map을 잘 복원합니다. Accuracy는 expert action과
argmax action이 같은 supervised sample의 비율이며 rollout SR을 의미하지 않습니다.

## 평가

평가는 기본적으로 GPU 한 장만 사용합니다. `max_episode_steps=128`에서 모든
agent가 목표에 최종 도착하면 **SR(success rate)=1**인 episode로 정의합니다.

- SR: 모든 agent가 성공한 episode의 비율(기존 CSR과 동일)
- ISR: 개별 agent 도착 비율(보조 지표)
- SoC: 각 agent의 최종 안정 도착 step 합
- Makespan: 마지막 agent의 최종 안정 도착 step
- Runtime: learned policy는 inference 합, MAPF-LNS2는 planning time

MAPF Transformer와 동일 expert data로 학습한 MAPF-GPT-6M을 공식 Random/Maze
scenario에서 비교:

```bash
cd pogema-mapf-transformer
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=../mapf-transformer-policy/src:src \
../MAPF/bin/python benchmark_compare.py \
  --suites random mazes \
  --models mapf_transformer mapf_gpt_6m \
  --agent-counts 8 16 24 \
  --map-limit 50 --seed-limit 1 \
  --current-checkpoint ../mapf-transformer-policy/runs/ablation_baseline_latent16/last.pt \
  --mapf-gpt-checkpoint ../mapf-gpt-mapf-lns2/runs/mapf_gpt_6m_mapf_lns2/last.pt \
  --device cuda:0 \
  --output-dir results/mapf_transformer_vs_mapf_gpt
```

MAPF-LNS2도 포함하려면:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=../mapf-transformer-policy/src:src \
../MAPF/bin/python benchmark_compare.py \
  --suites random mazes \
  --models mapf_transformer mapf_gpt_6m mapf_lns2 \
  --agent-counts 8 16 24 \
  --map-limit 50 --seed-limit 1 \
  --current-checkpoint ../mapf-transformer-policy/runs/ablation_baseline_latent16/last.pt \
  --mapf-gpt-checkpoint ../mapf-gpt-mapf-lns2/runs/mapf_gpt_6m_mapf_lns2/last.pt \
  --device cuda:0 \
  --mapf-lns2-binary external/MAPF-LNS2/lns \
  --mapf-lns2-cutoff 10 \
  --mapf-lns2-max-iterations 1000000000 \
  --mapf-lns2-workers 24 \
  --output-dir results/three_model_comparison
```

`episodes.csv`는 episode 완료 때마다 append되므로 같은 명령으로 resume할 수
있습니다. `summary.csv/json`에는 model, map family, agent 수별 aggregate가
저장됩니다. 양쪽 모두 SR=1인 공통 episode에서만 SoC/makespan/runtime을 비교해야
경로 품질 비교가 공정합니다.

그래프 생성:

```bash
cd pogema-mapf-transformer
../MAPF/bin/python plot_trained_model_comparison.py \
  results/three_model_comparison
```

이 명령은 같은 `episodes.csv`에서 MAPF Transformer와 MAPF-GPT-6M을 골라
map/agent별 SR 및 두 모델 모두 성공한 episode의 paired metric 그래프를 PNG/SVG로
저장합니다. MAPF-LNS2를 포함한 3-model 보고서는 서로 동일한 scenario로 실행한
learned-model 결과 directory와 LNS2 결과 directory를
`plot_three_model_comparison.py`에 전달해 생성할 수 있습니다.

단일 rollout을 SVG로 저장할 수도 있습니다.

```bash
cd pogema-mapf-transformer
python inference.py \
  --config configs/evaluation.yaml \
  --checkpoint ../mapf-transformer-policy/runs/ablation_baseline_latent16/last.pt \
  --episodes 1 --save-svg-dir renders
```

## Metadata/Graph ablation과 1-hour packed cache

기존 baseline은 유지하며 같은 1-hour trajectory로 다음 ablation을 실행할 수 있습니다.

- Experiment 1: grouped/gated metadata encoder
- Experiment 2: baseline metadata + edge-aware graph attention 1 layer
- Experiment 3: grouped metadata + edge-aware graph attention 1 layer

세 실험은 동일한 입력 필드를 사용하므로 packed cache는 한 번만 생성해 공유합니다.

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

예를 들어 Experiment 1의 2-GPU 학습은 다음과 같습니다.

```bash
mkdir -p mapf-transformer-policy/runs/packed_experiment1_grouped_metadata
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=mapf-transformer-policy/src \
torchrun --standalone --nproc_per_node=2 mapf-transformer-policy/train.py \
  --config mapf-transformer-policy/configs/packed_experiment1_grouped_metadata.yaml \
  2>&1 | tee mapf-transformer-policy/runs/packed_experiment1_grouped_metadata/console.log
```

Experiment 2와 3은 각각
`packed_experiment2_graph_attention_l1.yaml`,
`packed_experiment3_grouped_metadata_graph_l1.yaml`을 사용합니다. Packed cache는
lossless cache이며 GPU에서 기존 입력 tensor로 복원되므로 raw/packed 입력에 따른
모델 의미와 학습 target은 동일합니다.

## 테스트

```bash
PYTHONPATH=mapf-transformer-policy/src MAPF/bin/python -m pytest \
  mapf-transformer-policy/tests

PYTHONPATH=mapf-transformer-policy/src:pogema-mapf-transformer/src \
MAPF/bin/python -m pytest pogema-mapf-transformer/tests
```

Packed pipeline은 synthetic/실제 MAPF-LNS2 episode에서 raw 입력과 모든 tensor,
logit, total/action/map loss가 일치하는지 검증했습니다. Model forward/backward,
checkpoint 저장·복원, stable tracking, collision 검증도 test에 포함됩니다.

## 참고 문서

- [데이터 생성 전용 환경](DATA_GENERATION_README.md)
- [Packed cache 상세](mapf-transformer-policy/PACKED_DATASET.md)
- [Policy 구조 및 API](mapf-transformer-policy/README.md)
- [POGEMA 연동 및 benchmark](pogema-mapf-transformer/README.md)
- [MAPF-GPT trajectory 변환](mapf-gpt-mapf-lns2/README_CONVERSION.md)

## License 및 출처

이 저장소의 코드는 각 하위 project의 license를 따릅니다. POGEMA, MAPF-GPT,
MAPF-LNS2 map catalog 및 solver는 각각의 upstream license를 따르며, MAPF-LNS2
binary는 사용자가 공식 source에서 직접 build해야 합니다.
