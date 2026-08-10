# C++ Feature Generator 변경 범위

## 유지 사항

- Structured-25 map encoder와 기존 MPCT 모델 파라미터
- 256-token layout과 모든 prediction head
- 학습 및 checkpoint 형식
- CS-PIBT resolver의 입출력 계약

## 변경 사항

- `EpisodeFeatureBuilder`의 Python runtime 경로를 상태형 C++ generator로 교체
- static map과 goal별 CTG distance map을 episode 동안 캐시
- 최근 5-step history를 증분 갱신
- 모든 Ego의 local map, visible-agent slot, candidate feature를 한 번에 생성
- Python 기준 구현과 모든 `PolicyBatch` tensor를 필드별 비교하는 parity test 추가

## 성능 검증

- Feature 생성
- Batch 조립 및 CPU→GPU 전송
- Model forward
- Resolver

위 네 구간을 16/24/32 Agent 조건에서 각각 측정한다.
