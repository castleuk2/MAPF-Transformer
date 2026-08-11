# Implementation Notes

## 핵심 파일

| 파일 | 역할 |
|---|---|
| `src/mapf_sst/map_encoder.py` | Structured 25 Map Encoder |
| `src/mapf_sst/tokenizers.py` | Current-Agent 8-token, History 4-token encoding |
| `src/mapf_sst/spatial_fusion.py` | deterministic source/target gather + sparse map attention |
| `src/mapf_sst/structure_bias.py` | semantic/time/track/conflict/map attention bias |
| `src/mapf_sst/model.py` | 256-token encoder와 Ego Candidate 직접 scoring |
| `src/mapf_sst/communication.py` | optional real cross-view message exchange |
| `src/mapf_sst/data/npz_dataset.py` | Ego recentering, current/history selection, grouped all-view construction |
| `src/mapf_sst/losses.py` | Ego-only CE, optional ranking/map/semantic reconstruction |
| `train.py` | communication 없는 token baseline 학습 |
| `train_grouped.py` | all-Ego grouped communication 학습 |
| `inference.py` | 단일 Ego preference 출력 |
| `inference_grouped.py` | 모든 Agent의 own-view message-conditioned preference 출력 |

## 이전 구조에서 제거한 요소

- Entity summary token
- Recent/Trend history summary
- Top-24 Relation token
- Event token
- feature-packed Scene token
- ACT request token
- 다른 Ego 관점에서의 Neighbor action CE

## checkpoint 호환성

새 구조는 token 위치와 의미가 변경되었으므로 기존 v1/v2 checkpoint와 호환되지 않음. `checkpoint.py`는 format version 3만 허용하여 의미가 다른 checkpoint를 조용히 불러오는 문제를 방지함.
