# MAPF Preference–Coordination Transformer — C++ Feature Generator

이 디렉토리는 기존 MPCT 모델 구조와 checkpoint 호환성을 유지하면서,
Python 기반 Ego feature 생성을 상태형 C++ generator로 교체하기 위한 독립 개발본이다.
원본은 상위의 `MAPF_Preference_Coordination_Transformer_Project`에 그대로 보존한다.

상태형 C++ Runtime generator가 구현되어 있으며 Python `PolicyBatch` 기준 구현과
필드별 parity test를 수행한다. 학습 Dataset builder는 재현성을 위해 Python 기준 구현을 유지한다.

```python
from mapf_pct.runtime import PreferenceCoordinationPolicy

policy = PreferenceCoordinationPolicy(
    "runs/model/best.pt",
    device="cuda:0",
    feature_backend="cpp",
)
```

Feature 생성 속도는 다음 명령으로 비교할 수 있다.

```bash
PYTHONPATH=src python benchmark_feature_generators.py --steps 30 episode_n16.npz episode_n24.npz episode_n32.npz
```

25개의 Structured Map token과 25명 agent의 구조화된 Entity–History–Action token을 결합하여, expert MAPF action preference와 다중-agent coordination을 학습하는 PyTorch 연구 프로젝트이다.

## 핵심 구조

```text
17×17 Local Map
  └─ 5×5 structured 3×3 patches ─────────────── 25 Map tokens

25 Agents
  └─ Entity ×1 + History ×2 + Candidate ×5 ── 200 Agent tokens

Top-risk agent pairs ─────────────────────────── 24 Relation tokens
Past group events ─────────────────────────────── 5 Event tokens
Global operating regime ───────────────────────── 1 Scene token
Ego decision query ─────────────────────────────── 1 ACT token
                                                     ---------
                                                     256 tokens
```

`Candidate-to-Map Cross-Attention → Explicit Relation Construction → 4× Bidirectional Coordination Blocks → 25×5 Action Preference → CS-PIBT-style Resolver` 순서로 동작한다.

## Scene Token

Scene token은 특정 agent의 정보를 압축하는 token이 아니라, **장면 전체의 혼잡·진행·교착 위험 및 문제 설정을 나타내는 global workspace**이다.

초기 Scene token에는 obstacle/agent 비율, hop 통계, 최근 WAIT·override 비율, candidate 경쟁·edge-swap·bottleneck 비율, congestion, history coverage, one-shot/lifelong 및 stay/disappear mode가 들어간다. 이후 모든 token과 양방향 attention하여 전역 문맥을 집약하고 candidate·relation token에 다시 전달한다.

ACT token과의 차이는 명확하다.

- Scene token: “현재 장면이 어떤 coordination regime인가?”
- ACT token: “그 장면에서 ego가 지금 무엇을 해야 하는가?”

자세한 내용은 [`docs/SCENE_TOKEN.md`](docs/SCENE_TOKEN.md)를 참조한다.

## Structured Map Encoder

첨부 실험 보고서의 Structured 25 설계를 코드화하였다.

- 입력: `17×17 = 15×15 Core + 1-cell Halo`
- 25개 고정 3×3 patch
- patch feature: `9×2 cell one-hot + 4×3 opening + 4 outer-edge = 34-D`
- patch row/column 및 center embedding
- relative-position bias와 connectivity bias
- 공유 9-cell reconstruction decoder

보고서의 reconstruction 결과는 구조적 patch binding의 효율성을 뒷받침하지만, reconstruction-only 성능이 closed-loop MAPF 성공률을 보장하지 않으므로 이 프로젝트는 action, conflict, reason 및 rollout 평가를 별도로 둔다.

## 설치

```bash
python -m pip install -e ".[dev]"
```

의존성이 이미 설치된 폐쇄망 환경에서는 다음처럼 build isolation을 끌 수 있다.

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

## 빠른 검증

```bash
python examples/inspect_token_layout.py
python examples/run_synthetic.py
pytest
```

Tiny synthetic smoke training:

```bash
python train.py --config configs/tiny.yaml --max-steps 2
```

Inference:

```bash
python inference.py --checkpoint runs/tiny/best.pt --seed 7
```

Checkpoint 없이 실행하면 random-init 구조 검증만 수행한다.

## Expert NPZ 학습

기존 baseline workspace와 유사한 episode 형식을 지원한다.

```text
obstacles [H,W]
positions [T+1,N,2]
goals     [N,2] or [T+1,N,2]
actions   [T,N]
```

Manifest의 각 줄은 NPZ 경로 문자열 또는 `{"path": "..."}` 형식이다.

```bash
cp configs/npz_example.yaml configs/my_experiment.yaml
# manifest 경로와 coordinate_order 수정
python train.py --config configs/my_experiment.yaml
```

## 주요 출력

- `all_agent_logits`: `[B,25,5]`
- `ego_logits`: `[B,5]`
- `conflict_logits`: `[B,24,4]`
- `reason_logits`: `[B,25,12]`
- `scene_risk_logits`: `[B,3]`
- optional map reconstruction: `[B,15,15,2]`

## Loss

```text
Ego action CE
+ 0.35 × valid-neighbor action CE
+ 0.5 × ACT request CE
+ 0.02 × map reconstruction CE
+ 0.10 × relation conflict CE
+ 0.10 × reason multi-label BCE
+ 0.05 × scene-risk CE
```

실험에서는 각 weight를 ablation해야 한다. 특히 neighbor GT는 자체 ego-view sample에서는 1.0으로 둘 수 있지만, 하나의 ego 관측 안에 있는 neighbor action은 부분 관측 label noise 때문에 기본 0.35로 설정하였다.

## 구현 범위와 한계

- Synthetic dataset은 shape/backward/checkpoint 검증용이며 expert 성능을 의미하지 않는다.
- NPZ adapter의 reason은 heuristic label이다. Expert solver 내부 reason 또는 counterfactual cost가 있으면 대체해야 한다.
- 제공 resolver는 priority inheritance와 backtracking을 포함한 단일-step CS-PIBT-style 구현이다. 공식 외부 CS-PIBT와 동일하다고 간주하면 안 된다.
- 현재 architecture의 실제 우수성은 agent 수별 SR, SoC, makespan, deadlock, oscillation, WAIT ratio, resolver modification rate로 검증해야 한다.

## 참고 구현

- Baseline workspace: `https://github.com/castleuk2/MAPF-Transformer/tree/baseline-original`
- 본 프로젝트는 해당 repository를 복제한 것이 아니라, 공개된 I/O 관례와 이번 설계 논의를 바탕으로 작성한 독립 연구 scaffold이다.
