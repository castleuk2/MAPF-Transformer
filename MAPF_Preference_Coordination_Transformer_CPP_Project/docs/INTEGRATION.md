# 기존 MAPF-Transformer / POGEMA 데이터와의 연결

## 지원 NPZ 형식

`EpisodeFeatureBuilder`는 다음 배열을 읽는다.

```text
obstacles : [H,W], 0 free / 1 blocked
positions : [T+1,N,2]
goals     : [N,2] 또는 [T+1,N,2]
actions   : [T,N], WAIT/UP/DOWN/LEFT/RIGHT = 0..4
```

`coordinate_order`는 반드시 확인해야 한다.

- `row_col`: 배열 좌표가 그대로 `(row, col)`
- `xy`: 입력 `(x, y)`를 내부 `(row, col)`로 교환

## Ego 재중심화

하나의 supervised item은 `(episode, time step, ego agent)`이다. 현재 ego를 slot 0으로 놓고, 현재 15×15 Core 안의 가까운 agent를 최대 24명 선택한다. 선택한 global agent ID는 동일 sample의 5-step history에서 고정하여 feature binding을 유지한다.

## 학습 GT

- 각 agent를 ego로 재중심화한 sample에서는 해당 ego action을 primary GT로 사용한다.
- 동일 sample의 neighbor action도 auxiliary GT로 사용한다.
- 기본 loss weight는 ego 1.0, neighbor 0.35이다.

## 주의사항

1. 제공된 reason label은 expert 내부 reasoning이 아니라 관측 가능한 factor로 만든 heuristic multi-label이다. 진짜 expert reason이 있으면 교체해야 한다.
2. `CSPIBTResolver`는 단일 step의 deterministic resolver 역할을 구현한 연구용 코드이며 외부 공식 구현과 bit-exact 동일성을 주장하지 않는다.
3. local inference 예제의 resolver는 15×15 local domain에서 작동한다. 실제 joint rollout에서는 global obstacle map과 global positions를 전달해야 한다.
4. Structured Map Encoder checkpoint를 이식하려면 parameter key와 34-D feature 순서를 먼저 일치시켜야 한다.
