# Preference–Coordination Transformer 구조

## 고정 256-token context

```text
25 Structured Map Patch Tokens
25 × [1 Entity + 2 History + 5 Action Candidates] = 200 Agent Tokens
24 Explicit Top-risk Agent-pair Relation Tokens
5 Past Event Tokens
1 Scene Token
1 ACT Request Token
---------------------------------------------------
256 Tokens
```

## 처리 순서

1. `17×17 Local Map`에서 중앙 `15×15 Core`를 25개의 고정 `3×3` patch로 분할한다.
2. 각 patch를 `18 cell one-hot + 12 opening + 4 outer-edge = 34-D`로 인코딩한다.
3. Relative-position 및 connectivity attention bias를 사용하여 25개 Map token을 처리한다.
4. 각 agent를 `Entity + Recent History + Trend History + 5 Candidates`로 인코딩한다.
5. 125개 Candidate token이 target patch를 기준으로 25개 Map token을 cross-attention한다.
6. 현재 위치, candidate target overlap, edge swap, bottleneck, congestion으로 300개 가능한 pair 중 상위 24개 Relation token을 선택한다.
7. Map, Agent, Relation, Event, Scene, ACT token을 4개 bidirectional Coordination block에서 처리한다.
8. 공유 Candidate head가 `25×5` preference logits를 출력한다.
9. ACT head가 ego-specific residual logits를 출력한다.
10. CS-PIBT-style resolver가 full preference distribution을 이용해 vertex conflict와 edge swap을 제거한다.

## 구조적 binding

Map token은 특정 3×3 patch에, Agent token은 stable local slot에, Candidate token은 특정 `(agent, action)`에, Relation token은 특정 agent pair에 연결된다. 익명 latent로 모든 agent 정보를 조기에 혼합하지 않는다.
