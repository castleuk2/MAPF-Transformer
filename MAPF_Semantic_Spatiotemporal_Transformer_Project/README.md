# NovaTek MAPF Semantic Spatiotemporal Transformer

이 프로젝트는 기존 `Entity + Recent/Trend History + Relation/Event/Scene/ACT` 중심 256-token 구조를 대체하는 연구용 PyTorch 구현임. 핵심 목표는 **서로 다른 의미의 feature를 하나의 summary token에 조기 압축하지 않고**, Map·현재 Agent·과거 시점 정보를 고정 semantic slot으로 보존하여 Ego Agent의 5-action preference를 예측하는 것임.

## 1. 최종 256-token 구조

```text
25 Structured Map
+ 14 × 8 Current-Agent
+ 7 × 4 × 4 Factual History
+ 1 Self-Message Query
+ 6 Neighbor Messages
= 256 Tokens
```

기본 tokenization 실험에서는 마지막 7개 coordination slot을 모두 mask함. 실제 Agent 간 learned communication을 평가할 때만 `Self-Message Query 1개 + Neighbor Message 6개`로 활성화함.

### 고정 index

```text
0   – 24  : Structured Map Tokens                         25
25  – 136 : Current Agents 14 × 8                       112
137 – 164 : History t-1, 7 tracks × 4                    28
165 – 192 : History t-2, 7 tracks × 4                    28
193 – 220 : History t-3, 7 tracks × 4                    28
221 – 248 : History t-4, 7 tracks × 4                    28
249       : Self-Message Query                            1
250 – 255 : Neighbor Messages                             6
```

## 2. Current-Agent token

Agent당 8개 token을 사용함.

```text
[P, G, R, C_WAIT, C_UP, C_DOWN, C_LEFT, C_RIGHT]
```

- `P`: Ego 기준 Local X/Y
- `G`: Agent 기준 Relative Goal X/Y
- `R`: Remaining hops와 reachable 상태
- `C_*`: 방향별 one-hop CTG, ΔCTG, greedy 여부, static feasibility 등 해당 action 자체의 정보

Current slot은 Ego를 0번에 고정하고, 15×15 core 안에서 Ego와의 Manhattan
거리가 가까운 Agent부터 최대 13명을 선택함. 동일 거리에서는 global Agent ID가
작은 순서로 고정하여 Python/C++ generator가 동일한 slot을 생성함.

Source/target patch 위치는 token feature로 다시 학습시키지 않고 코드에서 계산함. Static obstacle과 out-of-bound는 hard action mask로 처리함.

## 3. Factual History token

Ego와 현재 Manhattan 거리가 가까운 최대 6개 Neighbor를 포함하여 7개 track을
선택하고, `t-1`부터 `t-4`까지 동일 Agent의 원본 의미를 유지함. 즉 History는
Current의 거리 정렬 결과 앞 7개 slot을 그대로 사용함.

```text
[P^τ, G^τ, R^τ, AO^τ]
```

- `P^τ`: 해당 과거 시점 위치
- `G^τ`: 해당 시점의 Relative Goal
- `R^τ`: 해당 시점의 Remaining hops
- `AO^τ`: 선택한 행동과 관측된 실제 이동 결과를 결합한 Action–Outcome token

과거 시점마다 5개 counterfactual candidate를 다시 만들지 않으며, Recent/Trend pooling도 사용하지 않음.

## 4. Map encoder

Map 부분은 별도 재구현을 사용하지 않음. `mapf-structured-map-transformer` 패키지의
`StructuredMapTransformer`를 직접 소유하는 adapter를 사용하고, 그 프로젝트의 native
checkpoint를 `strict=True`로 불러온 뒤 모든 Map parameter를 Freeze함. Freeze 상태에서는
policy 전체가 `train()` mode여도 Map Encoder는 `eval()`을 유지하므로 Dropout도 비활성화됨.

`StructuredMapTransformer`는 17×17 관측의 중앙 15×15 Core를 5×5개의 비중첩 3×3 patch로 분할함.

각 patch raw feature는 다음 34차원임.

```text
9 cells × Free/Obstacle one-hot = 18
4 directions × 3-bit opening   = 12
Outer-edge                      = 4
Total                           = 34
```

Shared MLP, row/column/center encoding, relative-position bias, connectivity bias를 거쳐 25개의 고정 공간 token을 생성함. 선택적으로 15×15 map reconstruction head를 제공함.

기본 config가 사용하는 checkpoint:

```text
../mapf-structured-map-transformer/runs/policy_exposure_structured_25_ce/best.pt
```

Frozen Map reconstruction은 optimizer를 갱신하지 않으므로 기본 학습의 `map_loss_weight`는 0임.

## 5. Candidate–Map 결합

`SpatiallyRoutedCandidateMapFusion`은 일반적인 전역 Agent→Map cross-attention 대신 다음의 hybrid 방식을 사용함.

1. source patch token을 deterministic gather
2. target patch token을 deterministic gather
3. source/target patch 내부 cell slot embedding 추가
4. source/target 주변 patch에만 sparse local cross-attention
5. 이후 전체 spatiotemporal Transformer에서 다시 통합

이미 알고 있는 공간 대응은 코드로 보장하고, attention은 주변 topology를 학습하도록 역할을 제한함.

## 6. Agent 관계

명시적 Top-K Relation Token을 사용하지 않음. 가능한 모든 current candidate pair에 다음 bias를 직접 적용함.

- same-agent block bias
- same-track temporal bias
- relative-time bias
- vertex target overlap bias
- edge-swap bias
- occupied-target bias
- corridor/bottleneck competition bias
- candidate↔source/target map patch bias

따라서 24개의 Relation slot 제한이나 24위/25위의 Top-K 불연속이 없음.

전체 256개 token에는 semantic field embedding과 별도로 학습 가능한 absolute
position embedding을 더함. 따라서 같은 종류의 field라도 sequence 내 고정 slot
(Map patch, Agent 번호, history lag, candidate 방향)을 명시적으로 구별할 수 있음.

## 6-1. 8-layer nearest-agent baseline

`configs/nearest_8layer_position_6epoch.yaml`은 다음 변경을 묶은 Comm-0 실험 설정임.

- spatiotemporal Transformer 8 blocks
- Current/History Agent를 Ego 기준 Manhattan 최근접 순으로 선택
- 256개 고정 slot의 learned absolute position embedding
- 기존 Structured Map encoder strict load 및 Freeze 유지

2 GPU 학습:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_ddp.py \
  --config configs/nearest_8layer_position_6epoch.yaml
```

### Factorized-track 수정 구조

`configs/nearest_8layer_factorized_track_6epoch.yaml`은 기존 실행과 checkpoint를
보존하는 별도 ablation임. 256개 slot마다 독립적인 absolute embedding을 주는 대신
공간·시간·identity의 의미에 맞게 위치 표현을 분리함.

```text
Map       : Frozen Structured Map encoder의 2-D patch position
Current   : Ego-relative coordinate + semantic field + shared track
History   : current-Ego-frame coordinate + lag + semantic field + shared track
Candidate : action field + source/target patch/cell
Message   : message slot + source-agent slot
```

같은 Current Agent와 그 Agent의 네 History 시점은 동일한 track embedding row를
공유함. Token에 직접 더하는 field·role·lag·track·cell embedding은 모두 표준편차
0.02로 초기화하여 특정 embedding 종류가 합산값을 지배하지 않게 함.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_ddp.py \
  --config configs/nearest_8layer_factorized_track_6epoch.yaml
```

## 7. 행동 출력

Scene Token과 ACT Token을 사용하지 않음. Transformer 이후 Ego의 고정 5개 Candidate Token에서 직접 scalar score를 생성함.

```text
C_WAIT, C_UP, C_DOWN, C_LEFT, C_RIGHT
        ↓ shared score head
5-action logits
        ↓ softmax
Ego action preference
```

프로젝트는 특정 CS-PIBT/PIBT/LaCAM backend를 전제하지 않음. `IndependentArgmaxBackend`는 인터페이스 예시일 뿐이며 collision-free를 보장하지 않음. 외부 joint-action backend는 `JointActionBackend` protocol로 연결할 수 있음.

## 8. 학습 GT 원칙

모든 Agent의 expert action을 활용하되, **각 Agent를 자신의 Ego-centered observation으로 재중심화한 sample에서 해당 Agent 자신의 action만 primary GT로 사용**함.

```text
A의 Ego view → A의 action GT
B의 Ego view → B의 action GT
...
```

다른 Ego 관점에서 Neighbor action을 CE로 감독하지 않음. grouped 학습에서는 한 frame의 모든 Ego view를 한꺼번에 구성하여 all-agent self-view loss를 평균할 수 있음.

## 9. Optional learned communication

단순히 여러 Ego view를 batch로 처리하면 sample 간 attention은 발생하지 않음. `MultiRoundCommunicationPolicy`는 이를 명시적으로 해결함.

```text
1차 pass: 각 Ego view → Self Message Query → self message
message gather: 선택된 6개 Neighbor의 self message 수집
2차 이상 pass: 6 Neighbor Message Token으로 local context 재평가
최종: 각 Agent 자신의 Ego action preference 출력
```

`rounds=0`은 semantic-token baseline, `rounds=1/2/4`는 communication ablation에 사용함.

## 10. 설치

```bash
python3 -m pip install -e ../mapf-structured-map-transformer --no-deps
cd MAPF_Semantic_Spatiotemporal_Transformer_Project
python3 -m pip install -e ".[dev]"
```

이미 PyTorch와 의존성이 설치된 폐쇄망 환경에서는 다음처럼 설치할 수 있음.

```bash
python3 -m pip install -e . --no-build-isolation --no-deps
```

## 11. 구조와 smoke test

```bash
python3 examples/inspect_token_layout.py
python3 examples/run_synthetic.py
pytest
```

## 12. Baseline 학습

```bash
python3 train.py \
  --config configs/tiny.yaml \
  --max-steps 2
```

`tiny.yaml`은 64-D software smoke test라 256-D pretrained Map checkpoint와 호환되지
않으며, 임의 초기화된 원본 Map class를 사용함. 실제 학습용 `base.yaml`,
`npz_example.yaml`, `communication.yaml`은 모두 256-D pretrained Map을 strict load하고 Freeze함.

실제 NPZ episode를 사용하려면 `configs/npz_example.yaml`의 manifest 경로를 수정함.

```bash
python3 train.py --config configs/npz_example.yaml
```

## 13. Optional communication 학습

```bash
python3 train_grouped.py \
  --config configs/communication.yaml \
  --steps 100
```

### 동일 Expert Data · Communication Rounds=4 · 2 GPU · 3/6 Epoch

이 branch의 재현 설정은 `configs/communication_rounds4_6epoch.yaml`임. Train은
기존과 동일한 `mapf_lns2_1h/train_manifest.jsonl`, validation은 동일한 held-out
manifest의 Maze/Random × 16/24/32 각 100 frame, 총 600 grouped frame을 고정 사용함.
한 grouped frame은 해당 시점의 모든 Agent own-view를 함께 구성하므로, 임의의
65,536개 단일-Ego validation sample 대신 완전한 message graph를 평가함.

Rounds=4는 다음 5회 policy pass를 의미함.

```text
Pass 0: 각 Ego view에서 Self Message 생성
Pass 1: Neighbor message 수집 후 갱신
Pass 2: 갱신 message 재교환
Pass 3: 갱신 message 재교환
Pass 4: 최종 message-conditioned action logits
```

설치와 실행:

```bash
git clone -b mapf-semantic-spatiotemporal-transformer \
  https://github.com/castleuk2/MAPF-Transformer.git
cd MAPF-Transformer/MAPF_Semantic_Spatiotemporal_Transformer_Project

python -m pip install -e ../mapf-structured-map-transformer --no-deps
python -m pip install -e . --no-build-isolation --no-deps
python build_cpp_extension.py
python verify_setup.py --config configs/communication_rounds4_6epoch.yaml

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_grouped_ddp.py --config configs/communication_rounds4_6epoch.yaml
```

출력은 `runs/sst_communication_r4_6epoch`에 저장됨. Epoch 3과 6에서 각각
`best_epoch3.pt`, `last_epoch3.pt`, `best_epoch6.pt`, `last_epoch6.pt`를 보존함.
Rounds=4는 Transformer block을 총 5번 실행하지만, Map encoder·Current/History
tokenizer·spatial fusion·Structure Bias는 첫 pass 전에 한 번만 계산하여 재사용함.
단순히 여러 Ego view를 batch로 묶는 것과 달리 실제 cross-view message가 전달됨.

## 14. 학습 속도 최적화

초기 구현은 batch의 동적 current↔history 및 candidate↔map bias를 Python 이중
반복문으로 갱신했음. Batch 128에서는 GPU 한 장당 forward마다 약 9,856회의 scalar
CUDA 접근과 작은 indexing kernel이 발생했으며, 이것이 MPCT보다 현저히 느렸던 주원인임.

현재 구현은 의미와 수식을 유지하면서 다음을 적용함.

- 동적 structured bias를 batch tensor 연산으로 완전 벡터화
- 256-token dense attention을 PyTorch fused SDPA로 실행
- local-map crop, visible-agent 검색, cell degree를 NumPy 벡터 연산으로 변경
- candidate/visibility feature를 episode 단위로 cache
- SST 전용 C++ all-Ego feature/communication-graph generator
- Communication 5개 pass의 Map·Agent·History context와 Structure Bias 공유
- Agent 수가 다른 여러 frame을 flattened Ego batch로 결합하고 graph index offset 보정
- DDP graph를 static graph로 고정하고 100 step마다 실제 samples/s 기록

실제 학습 episode 24개에서 C++와 Python의 모든 input tensor 및 communication graph가
일치했으며 C++ all-Ego feature 생성은 11.09배 빨랐음. Cached 4-round inference는
동일 출력에서 2.89배 빨랐고, 2-GPU multi-frame forward/backward smoke test도 통과함.
동일 장비의 2-GPU, per-GPU batch 128 baseline probe에서
100 update를 약 10.7초, 2,390.6 samples/s로 처리했음. 짧은 속도 검사는 다음과 같음.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src:../mapf-structured-map-transformer/src \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_ddp.py --config configs/same_data_6epoch.yaml --max-updates 100
```

## 15. 추론

단일 Ego preference:

```bash
python3 inference.py \
  --config configs/base.yaml \
  --checkpoint runs/base/best.pt \
  --npz /path/to/episode.npz \
  --time-step 10 \
  --ego 3
```

모든 Agent의 own-view preference와 learned message exchange:

```bash
python3 inference_grouped.py \
  --config configs/communication.yaml \
  --checkpoint runs/communication/last.pt \
  --npz /path/to/episode.npz \
  --time-step 10 \
  --rounds 2
```

## 16. NPZ 형식

```text
obstacles : [H, W]
positions : [T+1, N, 2]
goals     : [N, 2] or [T+1, N, 2]
actions   : [T, N]
```

Action index는 다음 순서임.

```text
0 WAIT
1 UP
2 DOWN
3 LEFT
4 RIGHT
```

자세한 내용은 `docs/DATA_FORMAT.md`를 참고함.

## 17. 검증 범위

포함된 테스트는 다음을 확인함.

- 정확한 256-token layout
- 25 Structured Map token과 15×15 reconstruction shape
- Agent/History padding mask
- Action–Outcome과 lag 구분
- deterministic source/target map routing
- same-track bias
- batching만으로 communication이 발생하지 않음
- explicit message wrapper에서 cross-view 정보가 전달됨
- Ego-only action loss
- NPZ adapter와 all-Ego view 구성
- checkpoint format v3

이 검증은 구현·shape·gradient 검증임. MAPF Success Rate, Sum of Costs, makespan, deadlock, oscillation 또는 collision-free 성능을 의미하지 않음.
