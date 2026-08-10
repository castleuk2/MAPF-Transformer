# MAPF-GPT Strict Semantic Tokenization

This branch also provides a preference-coordination runtime that keeps the
same Strict Semantic 256-token model and sends every Ego's five-action logits
to one centralized CS-PIBT resolver:

```python
from mapf_gpt_strict.runtime import StrictSemanticMPCTPolicy

policy = StrictSemanticMPCTPolicy("runs/strict_semantic/best.pt", device="cuda:0")
actions = policy.act_global(obstacles, positions, goals)
```

The direct-action `StrictSemanticPolicy` remains available as the controlled
baseline. Both runtime paths now use the same stateful `StableSlotAllocator`
as offline feature generation: Ego is slot 0, visible identities retain their
slots across distance reordering, and missing identities are reserved for two
steps. The pretrained map encoder also remains in `eval()` whenever frozen, so
its dropout cannot reactivate after a parent `model.train()` call.

공유 대화의 최종 `Strict Semantic Tokenization v1`만 입력부에 적용하고, 그 이후 정책 처리는 원본 MAPF-GPT와 같게 유지한 독립 실험 프로젝트입니다.

## 256-token layout

| index | block | layout |
|---:|---|---|
| 0–24 | Structured map | 17×17 halo에서 생성한 고정 5×5 patch latent 25개 |
| 25–128 | Current agents | Ego 우선 최대 13명 × 8 tokens |
| 129–248 | History | stable track 최대 6명 × 과거 5시점 × 4 tokens |
| 249–255 | Readout/PAD | 고정 PAD 7개; 마지막 위치에서 action을 읽음 |

현재 agent block은 `[Position, RelativeGoal, RemainingHops, WAIT, UP, DOWN, LEFT, RIGHT candidate]`입니다. History block은 각 track과 각 시점에 대해 `[Position, RelativeGoal, RemainingHops, ActionOutcome]`입니다. 부족한 agent/track은 PAD로 채우며, agent가 많으면 caller가 Ego + 12 neighbors와 history 6 stable tracks를 선택해야 합니다.

`StableSlotAllocator`는 학습 dataset과 runtime 양쪽에서 Ego를 slot 0에 고정하고 관측된 Neighbor의 기존 slot을 유지합니다. 일시적으로 사라진 Agent의 slot은 기본 2 frame 동안 예약합니다. Current slot 0–5와 History track 0–5는 동일 Agent를 가리키며, slot 6–12는 현재만 보이는 추가 Neighbor입니다.

Agent/History는 16차원 연속 feature로 구성한 뒤 shared feature MLP로 256차원 token을 만듭니다. Map 0–24번은 `mapf-structured-map-transformer`를 그대로 사용해 `17×17 halo → 중앙 15×15 → 25개 3×3 patch → 34차원 feature → shared MLP → 위치 encoding → spatial self-attention`으로 만든 `[B,25,256]` token입니다.

## Vocabulary-free feature encoding

Position/Goal은 정규화 좌표, Hops는 정규화 거리와 unknown flag, Candidate는 `valid·delta_hops·target state one-hot`, Action–Outcome은 두 개의 6-state one-hot으로 표현합니다. 모든 feature는 16차원으로 zero-pad한 뒤 `Linear → GELU → Linear → LN`을 거치며, agent-slot/field/time-lag/role/valid/absolute-position embedding을 더합니다. 마지막 hidden state는 vocabulary projection 없이 `Linear(256,5)`로 직접 Action logits을 출력합니다.

```python
logits, loss = model(encoded_observation, targets, halo_maps=halo_maps)
```

기존 map checkpoint는 `model.load_map_encoder(path, freeze=True)`로 불러와 고정할 수 있습니다.

## MAPF-GPT와 동일하게 유지한 부분

- 연속 feature MLP token embedding + learned absolute position embedding
- `agent-slot + field + time-lag + Ego/neighbor role + valid` semantic embedding
- 8-layer, 8-head, 256-dimensional pre-LN Transformer
- causal mask가 없는 full self-attention
- FFN `256 → 1024 → GELU → 256`
- 학습 target은 index 255 하나이며 나머지는 `-1`
- 마지막 token hidden state에서 `Linear(256,5)`로 Action logits 직접 출력

Relation/Scene/ACT token, attention bias/mask, Candidate 전용 head는 넣지 않았습니다. 이것들을 추가하면 “토큰 구성만 변경”한다는 비교 조건을 벗어나기 때문입니다.

## quick check

```bash
python -m pytest -q
```

## Portable training

Prepare manifests from the shared expert episodes, then launch two-GPU DDP:

```bash
python prepare_dataset.py \
  --train-manifest /path/to/train_manifest.jsonl \
  --val-manifest /path/to/val_manifest.jsonl \
  --eval-manifest /path/to/eval_manifest.jsonl

torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --epochs 3 --batch-size 128 --workers 10 \
  --map-checkpoint /path/to/structured_map_best.pt \
  --output runs/strict_semantic_stable_slots
```

The existing Strict Semantic checkpoint was trained before stable slots and
deterministic frozen-map mode were enforced. Retrain before comparing the new
MPCT runtime; using that old checkpoint would introduce a token-slot training
and inference mismatch.
