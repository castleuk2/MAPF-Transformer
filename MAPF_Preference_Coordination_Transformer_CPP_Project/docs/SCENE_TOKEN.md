# Scene Token의 역할

## 1. 정의

Scene token은 특정 agent나 action을 대신하는 토큰이 아니라, **현재 ego-centered MAPF 장면 전체의 운영 상태(regime)를 표현하는 단일 전역 토큰**이다. 초기 입력은 추론 시점에 계산 가능한 scene-level 통계와 문제 설정으로 구성되며, Coordination Transformer 내부에서 모든 Map·Agent·Candidate·Relation·Event token과 양방향으로 상호작용한다.

초기화는 다음과 같이 구현된다.

```text
learned [SCENE]
+ MLP(16-D scene statistics)
+ task-mode embedding(one-shot / lifelong)
+ target-mode embedding(stay / disappear)
```

## 2. 이 프로젝트의 16-D scene statistics

1. 15×15 Core의 obstacle 비율
2. 유효 agent 비율
3. Goal 도착 agent 비율
4. 잔여 hop 평균
5. 잔여 hop 표준편차
6. 잔여 hop 최솟값
7. 잔여 hop 최댓값
8. 직전 step의 WAIT 비율
9. 선택 action과 실제 이동이 달랐던 비율
10. 정적으로 실행 가능한 candidate 비율
11. greedy candidate 비율
12. 경쟁자가 존재하는 candidate 비율
13. edge-swap 위험 candidate 비율
14. bottleneck candidate 비율
15. 평균 local congestion
16. 5-step history coverage

이 값들은 모두 현재 관측과 과거 실행 결과에서 계산할 수 있어야 한다. Expert가 선택한 현재 action, 현재 reason GT, 미래 trajectory, solver가 사후에 결정한 joint action은 입력하면 안 된다.

## 3. Transformer 내부 역할

### Global aggregator

Agent 25명과 candidate 125개를 각각 비교하는 것만으로는 장면 전체가 저밀도인지, 단일 병목 경쟁인지, 다중 교착 전조인지 파악하기 어렵다. Scene token은 여러 relation 및 event를 한 위치로 집약한다.

### Global broadcaster

다음 block에서는 Scene token에 모인 전역 문맥이 다시 각 candidate와 relation으로 전달된다. 따라서 동일한 `UP` candidate도 장면이 저밀도일 때와 corridor 대치 상태일 때 다른 의미로 해석될 수 있다.

### Regime / tie-breaking context

개별 agent의 CTG와 map 조건이 비슷한 경우, 혼잡도·대기 누적·도착 agent 비율·task mode와 같은 전역 조건이 양보 순서와 우회 선호를 결정하는 tie-breaking 정보가 된다.

### Auxiliary supervision anchor

Scene token의 최종 embedding으로 `LOW / MEDIUM / HIGH` scene-risk 분류를 수행한다. 이 auxiliary head는 필수 출력이 아니라, Scene token이 실제로 전역 충돌 상태를 학습하는지 확인하기 위한 진단용 supervision이다.

## 4. ACT Request Token과의 차이

| 토큰 | 핵심 질문 | 입력/출력 역할 |
|---|---|---|
| Scene | “현재 다중-agent 장면은 어떤 운영 상태인가?” | 전역 문맥을 집약하고 전체 토큰에 재분배 |
| ACT | “이 문맥에서 ego는 지금 어떤 action을 선택해야 하는가?” | 최종 ego action logits를 생성하는 decision query |

ACT token은 ego Entity token을 anchor로 초기화되지만 GT action은 포함하지 않는다. Scene token은 최종 action query가 아니므로 두 토큰을 합치지 않는다.

## 5. 필수 ablation

- Scene token 제거
- learned Scene token만 사용하고 16-D 통계 제거
- 통계만 사용하고 Transformer 내 Scene token 제거
- Scene-risk auxiliary loss 제거
- task/target mode embedding 제거

평가는 one-step accuracy보다 closed-loop SR, deadlock, oscillation, WAIT ratio, resolver modification rate를 중심으로 수행한다.
