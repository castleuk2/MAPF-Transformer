# Expert-system Factors

Expert 정보를 모두 token 값에 넣지 않고 다음과 같이 분리함.

## 입력 token

- Local Position
- Relative Goal
- Remaining hops / unreachable
- 방향별 one-hop CTG와 ΔCTG
- Greedy 여부
- Candidate static feasibility
- 과거 Position·Goal·Hops
- 과거 selected action과 observed movement

## Bias / mask

- same-agent
- same-track
- relative time
- vertex target overlap
- edge-swap
- occupied target
- corridor/bottleneck competition
- source/target map patch
- obstacle/out-of-bound action mask

## Auxiliary target 후보

현재 구현의 기본 loss에는 넣지 않았지만 다음을 확장할 수 있음.

- 5-action expert ranking 또는 soft preference
- forced-first-action cost increase
- future conflict
- conflict partner/type
- yield / hindrance / regret
- deadlock·oscillation·stagnation
- future occupancy

## 입력 금지

- 현재 예측해야 할 expert action
- 미래 위치
- inference에서 접근할 수 없는 global solver state
- 사후 설명만으로 생성한 reason
