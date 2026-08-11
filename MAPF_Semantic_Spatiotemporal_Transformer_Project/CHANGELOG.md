# Changelog

## 0.3.0

- Fixed 256-token semantic layout로 전환
- Structured Map 25 유지
- Current Agent를 P/G/R/5 Candidates로 분리
- History를 시점별 P/G/R/Action–Outcome으로 보존
- Entity/Relation/Event/Scene/ACT token 제거
- Top-K relation 대신 all-pair attention bias 적용
- Candidate source/target deterministic map routing 추가
- Ego Candidate 직접 action scoring 적용
- 1 self-message query + 6 neighbor message optional communication 추가
- all-agent own-ego-view supervision 지원
- checkpoint format v3 도입
