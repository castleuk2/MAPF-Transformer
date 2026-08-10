# 공유 대화 설계 분석

## 채택한 최종안

대화 중간에는 `25 Entity + 50 History + 125 Candidate + 24 Relation + 5 Event + Scene + ACT` 안과 여러 변형이 제시됐습니다. 그러나 이후 질문과 검토를 거치며 hard Top-K relation의 불안정성, Scene/ACT의 중복, 한 token에 이질적 feature가 과도하게 합쳐지는 문제가 지적됐습니다. 이 프로젝트는 대화 마지막의 Strict Semantic Tokenization만 채택합니다.

```text
Map       25 = 25 fixed patch slots
Current  104 = 13 agents × (P, G, R, Cw, Cu, Cd, Cl, Cr)
History  120 = 6 tracks × 5 lags × (P, G, R, AO)
PAD        7
Total    256
```

- `P`: ego-local position
- `G`: agent-to-goal relative position
- `R`: remaining shortest-path hops
- `C*`: action별 valid 여부, hop 변화, target 상태
- `AO`: 해당 과거 상태에서 선택한 action과 다음 상태에서 관측한 이동

History는 각 track에서 오래된 시점부터 `t-1` 순서로 입력합니다. 5개보다 짧으면 왼쪽을 PAD하여 `t-1`의 absolute slot이 항상 같습니다. 현재 agent slot 0은 반드시 Ego이고 나머지 12명 및 history 6 tracks의 선정/추적 정책은 tokenizer 외부에서 결정합니다.

## 의도적으로 제외한 것

- Relation/Scene/ACT/Event token
- relation Top-K 및 quota
- connectivity/relative-position attention bias
- field별 병렬 encoder와 Candidate 전용 readout
- all-agent auxiliary action loss

이들은 유용할 가능성이 있지만 MAPF-GPT의 attention 또는 output head까지 바꾸므로 이번 통제 실험에서는 제외했습니다.

## 원본과의 정확한 경계

원본 tokenizer의 256 scalar token ID 생성만 `SemanticTokenizer`로 교체합니다. 그 결과는 여전히 `[B, 256]` 정수 ID이며, 이후 token/absolute-position embedding, non-causal MHA, pre-LN residual blocks, GELU FFN, 마지막 위치 action readout은 원본 흐름입니다.

Map 25개는 `mapf-structured-map-transformer`의 연속 latent를 그대로 사용합니다. 나머지 231개 slot은 16차원 semantic feature를 shared MLP로 256차원화한 뒤 이어 붙입니다. Vocabulary와 weight tying은 사용하지 않으며 마지막 hidden state에 5-action linear head를 적용합니다. 이후 8-layer non-causal Transformer 처리는 MAPF-GPT와 같습니다.

## Agent 구별 방식

공유 대화의 구별 원칙은 전역 Agent ID를 feature로 넣는 것이 아니라 `(1) Ego를 current slot 0에 고정, (2) neighbor를 결정적 순서의 semantic slot에 배치, (3) 같은 Agent의 history를 stable track slot에 유지, (4) 필요하면 Ego/neighbor role 및 valid/status embedding을 더함`입니다.

현재 구현은 `StableSlotAllocator`로 1–3을 수행합니다. Ego는 slot 0, history가 있는 주요 Agent는 slot 0–5, 현재만 사용하는 추가 Neighbor는 slot 6–12입니다. 각 token에는 absolute sequence position 외에 agent-slot, field, time-lag, Ego/neighbor role, valid embedding을 더합니다. Global Agent ID는 모델 입력에 직접 넣지 않고 runtime의 correspondence 관리에만 사용합니다.
