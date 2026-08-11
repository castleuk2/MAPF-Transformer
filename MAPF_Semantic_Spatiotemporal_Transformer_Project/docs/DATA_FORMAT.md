# Data Format

## NPZ episode

```text
obstacles : uint8 [H,W], Free=0, Obstacle=1
positions : int64 [T+1,N,2]
goals     : int64 [N,2] or [T+1,N,2]
actions   : int64 [T,N]
```

`coordinate_order`는 `row_col` 또는 `xy`를 지원함.

## Sample 생성

하나의 sample은 다음임.

```text
(episode, time_step, ego_agent)
```

- current Agent: Ego + 위험도 기반 최대 13 Neighbor
- history: Ego + temporal relevance 기반 최대 6 stable tracks
- 과거 Agent가 해당 시점에 Ego 관측에서 보이지 않았다면 history token을 mask
- 각 Agent action은 자신의 Ego view에서만 primary GT로 사용

## Grouped sample

`EpisodeFeatureBuilder.build_all_views()`는 동일 frame에서 모든 Agent를 차례로 Ego로 재중심화하여 batch를 만들고, history Neighbor identity를 communication graph에 연결함.
