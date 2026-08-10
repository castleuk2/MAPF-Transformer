# MPCT with Stateful C++ Feature Generator

MAPF Preference–Coordination Transformer(MPCT)의 모델·학습 구조는 유지하면서, rollout inference의 모든 Ego feature 생성을 상태형 C++ generator로 가속한 프로젝트이다. 학습 Dataset builder는 기존 Python 구현을 사용하므로 기존 실험과 데이터 구성이 동일하며, 추론 시 `python`과 `cpp` backend를 선택할 수 있다.

## 전체 흐름

```text
17×17 Ego Local Map
  → 5×5 structured patches                         25 Map tokens

최대 25 Agents
  → Entity 1 + History 2 + Action Candidate 5    200 Agent tokens

Agent-pair risk                                    24 Relation tokens
최근 group event                                    5 Event tokens
Scene summary                                       1 Scene token
Ego decision query                                  1 ACT token
                                                   ----------------
                                                   256 tokens
  → 4-layer bidirectional coordination Transformer
  → Ego action preference: WAIT/UP/DOWN/LEFT/RIGHT
  → global CS-PIBT-style resolver
  → collision-free actions
```

Map encoder는 사전학습된 Structured-25 checkpoint를 불러와 고정한다. 각 3×3 patch는 `9×2 cell one-hot + 4×3 opening bit + 4 outer-edge bit = 34-D` feature를 공유 MLP로 256차원 token으로 변환한다.

## 저장소에 포함된 학습 자산

`mpct-cpp-feature-generator` 브랜치에는 다른 PC에서 동일 학습을 시작하는 데 필요한 다음 자산이 포함되어 있다.

- Train/Validation NPZ 7,727개: `../pogema-mapf-transformer/data/mapf_lns2_1h/`
- Train/Validation manifest
- Frozen Structured-25 map checkpoint: `checkpoints/structured_map_policy_exposure_best.pt`
- 1-GPU 및 multi-GPU 학습 코드
- Python/C++ parity test와 feature benchmark

Map checkpoint SHA256:

```text
4d3ebbb73f90f2bcf496fca328f5d20192194fca72e63c71a8307c0e335632b2
```

## 1. 다른 PC에서 설치

Linux와 Python 3.10 이상을 권장한다. C++ runtime을 빌드하려면 C++17 compiler가 필요하다.

```bash
git clone --branch mpct-cpp-feature-generator --single-branch \
  https://github.com/castleuk2/MAPF-Transformer.git
cd MAPF-Transformer/MAPF_Preference_Coordination_Transformer_CPP_Project

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

CUDA에 맞는 PyTorch를 먼저 설치한 후 프로젝트를 설치한다. PyTorch 설치 명령은 PC의 CUDA 환경에 맞춰 선택해야 한다.

```bash
# 예: PyTorch 공식 설치 명령으로 torch를 먼저 설치
python -m pip install -e ".[dev]"
```

Ubuntu에서 compiler가 없다면 다음 패키지가 필요하다.

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
```

## 2. 학습 자산 검증

다른 PC로 clone한 직후 다음 명령을 먼저 실행한다.

```bash
python verify_setup.py
```

검증 항목은 다음과 같다.

- Frozen map checkpoint 존재 여부와 SHA256
- Train/Validation manifest 존재 여부
- Manifest가 가리키는 모든 NPZ 존재 여부
- Map checkpoint의 strict model load
- Trainable/frozen parameter 수

마지막에 `portable setup: OK`가 출력되어야 한다.

## 3. Dataset 정의

기본 설정은 저장소의 다음 manifest를 사용한다.

```text
../pogema-mapf-transformer/data/mapf_lns2_1h/train_manifest.jsonl
../pogema-mapf-transformer/data/mapf_lns2_1h/val_manifest.jsonl
```

각 NPZ episode 형식은 다음과 같다.

```text
obstacles [H,W]       uint8
positions [T+1,N,2]   row, column
goals [N,2]           row, column
actions [T,N]         WAIT=0, UP=1, DOWN=2, LEFT=3, RIGHT=4
```

각 Ego는 도착 전 policy-exposure step 전체와 도착 후 WAIT suffix의 20%를 사용한다. Validation은 Maze/Random을 균등하게 나누고, 각 Map 유형 안에서 16/24/32 Agents를 거의 동일하게 구성하여 최대 65,536 sample을 평가한다.

## 4. 학습

### 단일 GPU

```bash
python train.py \
  --config configs/baseline_dataset_frozen_map.yaml \
  --device cuda:0 \
  --output-dir runs/mpct_portable_single_gpu
```

### GPU 2장

설정의 `training.batch_size: 256`은 global batch size이며 GPU 2장에서는 GPU당 128 sample을 처리한다.

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --config configs/baseline_dataset_frozen_map.yaml \
  --output-dir runs/mpct_portable_ddp
```

GPU 수를 바꾸려면 `batch_size`가 GPU 수로 나누어져야 한다. VRAM이 부족하면 global `batch_size`와 `val_batch_size`를 함께 줄인다.

### 출력

```text
runs/<experiment>/
├── best.pt
├── last.pt
├── metrics.jsonl
└── resolved_config.yaml
```

`metrics.jsonl`에는 epoch별 Train/Validation total loss, 세부 loss와 Ego action accuracy가 저장된다.

## 5. Loss

```text
L = 1.00 × Ego action CE
  + 0.35 × valid-neighbor action CE
  + 0.50 × ACT request CE
  + 0.02 × map reconstruction CE
  + 0.10 × relation conflict CE
  + 0.10 × reason multi-label BCE
  + 0.05 × scene-risk CE
```

Frozen map encoder의 파라미터는 갱신하지 않지만, 기존 MPCT 학습 계약을 유지하기 위해 reconstruction loss 값은 추적한다.

## 6. 테스트

```bash
pytest -q
python examples/inspect_token_layout.py
python examples/run_synthetic.py
```

`test_cpp_feature_generator.py`는 동일 상태에서 Python과 C++이 생성한 Map, Agent, History, Candidate, Event, Scene tensor를 필드별로 비교한다.

## 7. C++ Runtime 사용

```python
from mapf_pct.runtime import PreferenceCoordinationPolicy

policy = PreferenceCoordinationPolicy(
    "runs/mpct_portable_ddp/best.pt",
    device="cuda:0",
    feature_backend="cpp",
)
```

기준 Python backend는 다음과 같이 유지된다.

```python
policy = PreferenceCoordinationPolicy(
    "runs/mpct_portable_ddp/best.pt",
    device="cuda:0",
    feature_backend="python",
)
```

C++ backend는 최초 import에서 로컬 compiler로 확장을 빌드하고 이후 빌드 결과를 재사용한다. Generator는 static map, goal별 CTG, 최근 위치와 선택 action을 상태로 유지하고 모든 Ego feature를 한 번에 생성한다.

## 8. Python/C++ 속도 비교

```bash
PYTHONPATH=src python benchmark_feature_generators.py \
  --steps 30 \
  --output reports/cpp_feature_benchmark/result.json \
  ../pogema-mapf-transformer/data/mapf_lns2_1h/maze/val/episode_0000000_medium-mazes-eval-seed-10000_s0_n16.npz \
  ../pogema-mapf-transformer/data/mapf_lns2_1h/maze/val/episode_0000001_medium-mazes-eval-seed-10000_s0_n24.npz \
  ../pogema-mapf-transformer/data/mapf_lns2_1h/maze/val/episode_0000002_medium-mazes-eval-seed-10000_s0_n32.npz
```

Runtime 객체는 다음 네 구간의 step별 시간도 기록한다.

```python
policy.last_timing
policy.timing_summary()
```

```text
feature_s
batch_transfer_s
forward_s
resolver_s
```

## 9. 사용자 Dataset 사용

`configs/npz_example.yaml`을 복사하고 Train/Validation manifest 경로를 수정한다. 상대경로는 프로젝트 루트 기준으로 해석되므로 실행 디렉토리에 의존하지 않는다.

```bash
cp configs/npz_example.yaml configs/my_dataset.yaml
python train.py --config configs/my_dataset.yaml --device cuda:0
```

Manifest의 각 줄은 최소한 다음 값을 가져야 한다.

```json
{
  "path": "maze/train/episode_0000000.npz",
  "arrival_steps": [20, 31, 18],
  "time_steps": 45,
  "map_family": "maze",
  "num_agents": 3
}
```

`path`는 manifest 파일이 위치한 디렉토리를 기준으로 해석한다.

## 10. 주의 사항

- C++ generator는 rollout inference를 최적화하며 학습 Dataset 생성은 Python 기준 구현을 사용한다.
- 최대 입력은 Ego 포함 25 Agents이며 초과 Agent는 Ego와 가까운 순서로 절단된다.
- 제공 resolver는 CS-PIBT-style 구현이며 외부 공식 CS-PIBT와 bit-exact하다고 가정하지 않는다.
- Synthetic Dataset은 구조 검증용이며 실제 MAPF 성능 학습용이 아니다.
- Closed-loop 평가는 SR, 공통 성공 episode의 SoC/Makespan/Runtime과 resolver action 변경률을 함께 확인해야 한다.

자세한 Token 구조는 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), Scene Token은 [`docs/SCENE_TOKEN.md`](docs/SCENE_TOKEN.md), 통합 방법은 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)를 참고한다.
