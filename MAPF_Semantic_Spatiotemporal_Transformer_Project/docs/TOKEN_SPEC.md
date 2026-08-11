# Token Specification

## 1. 전체 budget

| 구간 | index | 수 |
|---|---:|---:|
| Structured Map | 0–24 | 25 |
| Current Agent | 25–136 | 112 |
| History t-1 | 137–164 | 28 |
| History t-2 | 165–192 | 28 |
| History t-3 | 193–220 | 28 |
| History t-4 | 221–248 | 28 |
| Self Message Query | 249 | 1 |
| Neighbor Message | 250–255 | 6 |

## 2. Current Agent block

Agent slot `s`의 시작 index:

```text
base = 25 + 8*s
```

| offset | semantic |
|---:|---|
| 0 | Position |
| 1 | Relative Goal |
| 2 | Remaining Hops |
| 3 | WAIT Candidate |
| 4 | UP Candidate |
| 5 | DOWN Candidate |
| 6 | LEFT Candidate |
| 7 | RIGHT Candidate |

Slot 0은 항상 Ego임. Neighbor slot rank 자체는 모델 feature로 사용하지 않음.

## 3. History block

History는 lag-major로 배치됨. Lag `k∈{1,2,3,4}`, track `h∈{0,…,6}`의 시작 index:

```text
base = 137 + (k-1)*28 + h*4
```

| offset | semantic |
|---:|---|
| 0 | Historical Position |
| 1 | Historical Relative Goal |
| 2 | Historical Remaining Hops |
| 3 | Historical Action–Outcome |

Track 0은 Ego임. 다른 track은 current Agent 중 최근 상호작용 중요도가 높은 stable identity를 사용함.

## 4. Action–Outcome

```text
AO^τ = MLP(concat(E_selected(a^τ), E_observed(m^τ)))
```

특정 resolver 이름이나 priority rule을 token에 넣지 않음. 선택 행동과 관측 이동만 표현함.

## 5. Coordination slots

- Baseline: 249–255 모두 padding mask
- Query-only pass: 249만 활성
- Message pass: 249와 valid Neighbor Message slot 활성

Message는 semantic token interaction 이후 생성되는 late-fusion state이므로 기존 Entity/Scene summary와 역할이 다름.
