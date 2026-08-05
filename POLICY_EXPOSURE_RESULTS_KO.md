# Structured 25-token Policy-exposure 학습·평가 결과

## 구현 검수와 수정

- 공유 설계대로 17×17 halo 입력, 중앙 15×15의 3×3 patch 25개, 34차원 patch feature,
  25-token full self-attention, patch별 복원 구조를 확인했다.
- 기존 구현의 weighted BCE+Dice는 M8/M16/M32 실험의 Loss와 달랐다. 공정 비교 경로에
  Free/Obstacle 2-class logits과 동일한 cell-wise mean CE를 추가했다.
- 15×15 tensor만 읽던 dataset 대신 global obstacle map과 agent position에서 정확한 17×17 halo를
  만드는 Policy-history loader를 추가했다.
- 최종 평가는 batch별 비율 평균이 아니라 전체 Cell의 TP/FP/FN/TN과 CE 합을 누적하는 micro 방식으로
  구현했다.

## 데이터

| Split | Episodes | Policy samples | 유효 Map 노출 | 고유 layout |
|---|---:|---:|---:|---:|
| Train | 6,988 | 3,465,840 | 9,584,884 | 244 |
| Validation | 739 | 349,487 | 1,575,432 | 128 |
| Evaluation | 739 | 357,468 | 1,615,012 | 130 |

Train–Validation–Evaluation 사이 obstacle layout SHA-256 중복은 모두 0이다. Validation의 중앙
15×15 crop 100개를 기존 bit-packed map과 대조해 100개 모두 완전히 일치했다.

## 학습 조건

- Token: 25×256
- Transformer: 8 heads, 1 layer, MLP ratio 4, dropout 0.1
- Loss: 중앙 15×15 모든 Cell의 2-class mean CE
- AdamW: learning rate 0.0003, weight decay 0.1, scheduler 없음
- Epochs: 3, policy sample batch size 512, seed 42
- Best checkpoint: epoch 2, optimizer step 20,380

## 최종 전수 평가

| Split / Map 유형 | Maps | Cells | Cell CE | Cell Acc. | Obstacle F1 | IoU | 완전 복원율 | 실패 Maps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val 전체 | 1,575,432 | 354,472,200 | 0.0000 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0 |
| Val Maze | 984,370 | 221,483,250 | 0.0000 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0 |
| Val Random | 591,062 | 132,988,950 | 0.0000 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0 |
| Eval 전체 | 1,615,012 | 363,377,700 | 0.0000 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0 |
| Eval Maze | 1,058,535 | 238,170,375 | 0.0000 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0 |
| Eval Random | 556,477 | 125,207,325 | 0.0000 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0 |

## 해석 제한

각 token의 입력 feature에 담당 3×3 patch의 9개 Cell 상태가 직접 들어가고, 256차원 token에서 같은
9개 Cell을 복원한다. 따라서 이 결과는 25개 공간 정렬 token이 occupancy를 손실 없이 전달할 수 있음을
보이지만, M8/M16/M32처럼 225 Cell을 8·16·32개의 learned latent query로 압축하는 bottleneck과 난이도가
같다는 뜻은 아니다. 모든 Map이 성공했으므로 이 checkpoint에는 정성적으로 제시할 실패 사례가 없다.
