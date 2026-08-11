# Validation Scope

## Frozen Structured-25 integration

SST의 Map 경로는 유사 재구현이 아니라 `mapf_map_transformer.StructuredMapTransformer`를
직접 사용함. Policy-exposure `best.pt`를 `strict=True`로 읽고, checkpoint에 저장된 Map
config와 SST의 17×17 / 15×15 / 3×3 / 25-token / 256-D 계약을 검사함. 별도로 읽은 원본
모델과 adapter 출력을 bit-for-bit 비교하며, 모든 Map parameter의 `requires_grad=False`와
policy 학습 중 Map module의 eval mode 유지도 검사함.

## 구현 검증

- fixed 256-token layout
- map/current/history/coordination index
- Structured 25 raw feature와 reconstruction tensor
- invalid Agent/History padding isolation
- lag별 Action–Outcome 분리
- source/target spatial routing
- same-track attention bias
- optional multi-round message gradient
- Ego-only CE와 semantic reconstruction loss
- NPZ adapter와 all-Ego grouped view
- checkpoint v3 round-trip

## 아직 검증하지 않은 성능

- POGEMA closed-loop Success Rate
- Sum of Costs / makespan
- collision, deadlock, oscillation
- 14 current Agent coverage의 충분성
- 7 history tracks와 4-step horizon의 최적성
- communication round 수
- external PIBT/LaCAM/resolver와의 결합 효과
- GPU latency와 memory

Smoke training 수치는 실행 가능성만 확인하며 정책 성능으로 해석하지 않음.

## 실행 결과

```text
pytest: 13 passed
baseline smoke training: 2 optimizer steps completed
communication smoke training: 2 optimizer steps completed
single-view inference: [1,5] preference generated
six-view grouped inference: [6,5] preferences generated after 2 rounds
```

세부 로그는 `validation/`에 저장되어 있음.
