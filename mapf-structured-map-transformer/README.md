# MAPF Structured Map Transformer

15×15 Ego-centered binary map을 **25개의 공간적으로 정렬된 3×3 patch token**으로 변환하고, reconstruction loss를 통해 25-token bottleneck이 free/occupied 구조를 충분히 보존하는지 평가하는 독립 실행형 PyTorch 프로젝트이다.

입력은 중앙 15×15와 한 셀 폭 halo를 포함한 17×17이다. 17×17 전체를 token화하지는 않는다. 중앙 15×15만 5×5 patch grid로 나누며, halo는 15×15 외곽에서 free space가 실제로 이어지는지를 계산하고 이동 시 17개 신규 셀만 갱신하기 위해 사용한다.

## 1. 최종 구조

```text
17×17 static occupancy window
  ├─ one-cell halo: outer continuation / rolling update
  └─ center 15×15 core
          │
          ▼
5×5 non-overlapping patches, each 3×3
          │
          ▼
Patch feature builder
  ├─ 9 cell states as one-hot vectors
  ├─ N/S/W/E 3-bit movement openings (12 bits)
  └─ outer-edge indicator (4 bits)
          │
          ▼
Shared Patch MLP: 34 → 256 → 256
          │
          ▼
Learned 2D row/column positional encoding
          │
          ▼
One Spatial Map Transformer encoder block
  ├─ full 25-token self-attention
  ├─ 2D relative-position bias
  ├─ patch-connectivity bias
  └─ FFN
          │
          ├──────────────► 25 × 256 structured map tokens
          │
          ▼
Shared reconstruction head: each token → 9 cell logits
          │
          ▼
15×15 occupancy reconstruction
```

출력 token 수는 다음과 같이 고정된다.

\[
N_{\mathrm{map}}=
\left(\frac{15}{3}\right)^2=25
\]

각 token \(\mathbf{m}_{u,v}\)는 항상 5×5 patch grid의 동일한 위치 \((u,v)\)에 대응한다. learned latent query를 이용한 225→25 soft pooling이 아니므로 token index의 공간적 의미가 변하지 않는다.

## 2. Patch feature

binary map에서 `0=free`, `1=occupied`를 기본값으로 사용한다. 각 3×3 patch의 9개 상태를 2-class one-hot으로 변환한다.

\[
9\times2=18
\]

각 patch edge의 3개 셀에 대해, 현재 셀과 한 칸 바깥 이웃 셀이 모두 free일 때 opening을 1로 정의한다.

\[
o^d_k=
\mathbb{I}[s_k=\mathrm{free}]
\land
\mathbb{I}[s_{k+d}=\mathrm{free}]
\]

네 방향에서 3개 opening을 사용하므로 12차원이다. 15×15 바깥에 있는 이웃은 17×17 halo에서 얻는다. 내부 patch 경계와 외곽 경계가 동일한 식으로 처리된다.

기본 feature 차원은 다음과 같다.

\[
18\;\text{(cell states)} + 12\;\text{(ports)} + 4\;\text{(outer-edge mask)}=34
\]

이를 shared MLP로 256차원에 임베딩한다.

## 3. Spatial Map Transformer

25개 patch embedding에 2D positional encoding을 더한다.

\[
\mathbf{x}_{u,v}^{(0)}=
\mathbf{g}_{u,v}+E_{\mathrm{row}}(u)+E_{\mathrm{col}}(v)
\]

기본 모델은 **encoder block 1개**만 사용한다. 한 번의 full self-attention으로 모든 25개 patch가 서로 직접 정보를 교환할 수 있다.

Attention score에는 2D 상대 위치와 인접 patch의 3-bit opening code가 추가된다.

\[
A_{ij}^{(h)}=
\frac{\mathbf{q}_i^{(h)\mathsf T}\mathbf{k}_j^{(h)}}{\sqrt{d_h}}
+B_{\mathrm{rel}}^{(h)}(\Delta u,\Delta v)
+B_{\mathrm{conn}}^{(h)}(d,c_{ij})
\]

`num_layers`는 configuration에서 0, 1, 2 등으로 바꿀 수 있지만 기본값은 1이다.

## 4. Reconstruction loss

각 최종 token에서 해당 3×3 patch의 9개 occupied logits를 직접 예측한다. 25개 결과를 원래 공간 순서로 결합하여 15×15 map을 복원한다.

기본 reconstruction loss는 weighted BCE와 soft Dice loss의 합이다.

\[
\mathcal{L}_{\mathrm{recon}}
=\lambda_{\mathrm{BCE}}\mathcal{L}_{\mathrm{WBCE}}
+\lambda_{\mathrm{Dice}}\mathcal{L}_{\mathrm{Dice}}
\]

장애물 경계 셀에는 추가 pixel weight를 적용한다. free 셀이 많은 데이터에서는 `occupied_pos_weight`로 occupied class를 더 크게 반영한다.

25개 token의 적절성은 loss 하나만으로 판단하지 않고 다음 지표를 함께 기록한다.

- occupied IoU, precision, recall, F1
- 1-cell tolerance boundary F1
- 3×3 patch exact reconstruction rate
- 전체 15×15 exact reconstruction rate
- Ego 중심에서 계산한 free-space reachability IoU
- 중앙 connected component가 네 외곽 방향까지 도달하는지에 대한 agreement

좁은 통로가 중요한 MAPF에서는 cell accuracy보다 `occupied_iou`, `boundary_f1`, `reachability_iou`를 우선 확인하는 것이 적절하다.

## 5. 17-cell rolling update와 latent reuse

상태 메모리는 15×15가 아니라 **17×17 halo window 전체**를 보관한다. 실제 Ego displacement가 한 칸이면 기존 window를 shift하고 신규 row 또는 column 17개만 삽입한다.

| 실제 이동 | 유지 셀의 local shift | 신규 strip |
|---|---|---|
| UP | 아래로 1칸 | 최상단 row 17개 |
| DOWN | 위로 1칸 | 최하단 row 17개 |
| LEFT | 오른쪽으로 1칸 | 최좌측 column 17개 |
| RIGHT | 왼쪽으로 1칸 | 최우측 column 17개 |
| WAIT / 이동 실패 | 변화 없음 | 없음 |

WAIT 또는 이동 실패이고 static map이 변하지 않았다면 model forward를 수행하지 않고 **이전 latent tensor를 그대로 반환**한다. 명령 action이 아니라 `moved`로 전달된 실제 이동 결과를 기준으로 갱신한다.

```python
result = runtime.step(
    Action.RIGHT,
    moved=True,
    incoming_strip=new_right_column,  # shape [17]
)

# WAIT 또는 이동 실패
result = runtime.step(Action.WAIT, moved=False)
assert result.reused
```

다수 Agent를 위한 `VectorizedMapTokenRuntime`은 바뀐 Agent window만 batch로 다시 encode한다.

## 6. 설치

```bash
python -m pip install -e .
```

개발 환경까지 설치하려면 다음을 사용한다.

```bash
python -m pip install -e '.[dev]'
```

## 7. 빠른 검증

```bash
pytest -q
python train.py --config configs/smoke.yaml
python demo_runtime.py --checkpoint runs/smoke/best.pt
python evaluate.py \
  --checkpoint runs/smoke/best.pt \
  --visualization runs/smoke/evaluation.png
```

기본 256차원 학습:

```bash
python train.py --config configs/map_reconstruction.yaml
```

주요 출력:

```text
runs/map_reconstruction/
├── resolved_config.yaml
├── metrics.jsonl
├── best.pt
├── last.pt
├── epoch_XXX.pt
├── summary.json
└── visualizations/
```

## 8. 외부 데이터 사용

`.npz` 파일은 다음 배열을 포함한다.

```text
halo_maps: uint8/int64 [N,17,17]
```

### Policy-exposure 공정 비교

기존 M8/M16/M32 Map Autoencoder와 같은 데이터 노출 및 Loss로 비교할 때는 다음 설정을 사용한다.

```bash
python train.py --config configs/policy_exposure_ce.yaml
python evaluate_policy_exposure.py \
  --checkpoint runs/policy_exposure_structured_25_ce/best.pt \
  --splits val eval \
  --output runs/policy_exposure_structured_25_ce/policy_exposure_metrics.json
```

이 경로는 Train history만 결정적으로 truncation하고, Val/Eval은 사용 가능한 전체 history를 최대 5까지
사용한다. 중앙 15×15의 각 Cell에 Free/Obstacle 2 logits을 출력하며 기존 실험과 동일한 cell-wise mean CE로
학습한다. 공유 설계의 weighted BCE+Dice 기본 경로는 별도로 유지된다.

데이터 생성 예제:

```bash
python scripts/create_npz_dataset.py --output-dir data/halo_maps
```

configuration을 다음처럼 바꾼다.

```yaml
dataset:
  kind: npz
  train_path: data/halo_maps/train.npz
  val_path: data/halo_maps/val.npz
```

## 9. 기존 MAPF policy와 결합

Map encoder 출력은 다음 shape이다.

```python
map_output = map_encoder(halo_maps)
map_tokens = map_output.latent_tokens  # [B,25,256]
```

Agent token이 Query, map token이 Key/Value가 되는 cross-attention adapter가 포함되어 있다.

```python
from mapf_map_transformer import AgentMapCrossAttention

fusion = AgentMapCrossAttention(d_model=256, num_heads=8)
map_aware_agents = fusion(
    agent_tokens,     # [B,A,256]
    map_tokens,       # [B,25,256]
    agent_xy,         # [B,A,2], local x/y in [-7,7]
)
```

Map token은 static geometry를 유지하고 Agent token만 map-aware하게 갱신한다. 세부 교체 절차는 `docs/INTEGRATION.md`를 참조한다.

## 10. 설계상 주의점

- 17×17 halo가 있어도 token화 대상은 중앙 15×15뿐이다.
- WAIT 재사용은 static geometry가 변하지 않는다는 전제에서만 유효하다.
- 다른 Agent의 dynamic occupancy를 static map channel에 합치지 않는 것이 좋다.
- Ego가 이동하면 3×3 patch alignment가 바뀌므로 25개 patch token은 다시 생성한다. 증분 갱신은 17×17 cell buffer에서 수행한다.
- Reconstruction 성능이 높더라도 최종 MAPF rollout 성능을 보장하지는 않는다. 이후 action success rate, collision rate, deadlock rate와 함께 검증해야 한다.

## 11. 파일 구성

```text
src/mapf_map_transformer/
├── geometry.py       # patch/port 생성, 17-cell rolling update
├── model.py          # patch MLP + 25-token Spatial Transformer + decoder
├── losses.py         # reconstruction loss
├── metrics.py        # occupancy/boundary/reachability metrics
├── runtime.py        # single/vectorized latent cache
├── fusion.py         # optional Agent→Map cross-attention
├── data.py           # synthetic/NPZ datasets
└── trainer.py        # training and evaluation loop
```

## Upstream context

이 프로젝트는 MAPF Transformer의 map bottleneck 실험을 독립적으로 수행하기 위한 모듈이다. Action order는 POGEMA 관례인 `WAIT, UP, DOWN, LEFT, RIGHT`를 사용한다.
