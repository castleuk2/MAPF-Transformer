# Model I/O schema

## Batched training input

All tensors use batch-first layout.

| Key | Shape | Type | Meaning |
|---|---:|---|---|
| `local_maps` | `B×15×15×15` | long | 15 frames of 15×15 free/blocked maps |
| `agent_x` | `B×15×16` | long | local X, 0–14; 15 is invalid sentinel |
| `agent_y` | `B×15×16` | long | local Y, 0–14; 15 is invalid sentinel |
| `action_mask` | `B×15×16×4` | float/bool | multi-hot UP/DOWN/LEFT/RIGHT shortest-path descent set |
| `distance` | `B×15×16` | long | six-bit distance bucket, 0–63 |
| `agent_valid` | `B×15×16` | bool | slot contains a visible agent |
| `track_reset` | `B×15×16` | bool | a new identity was assigned to the stable slot |
| `previous_action` | `B×15` | long | POGEMA action 0–4; 5 is START |
| `actual_move` | `B×15` | long | action implied by observed displacement |
| `outcome` | `B×15` | long | START/SUCCESS/FAILED/WAIT |
| `visible_count` | `B×15` | long | valid Ego + neighbors, 0–16 |
| `frame_valid` | `B×15` | bool | false for leading PAD frames |
| `target` | `B` | long | expert current action, 0–4 |

Agent slot order is neighbor slots 0–14 and Ego slot 15. Each valid frame
becomes 16 map-conditioned agent tokens plus one transition token.

## Physical payload

The compact storage word uses 18 meaningful bits inside `uint32`:

- bits 0–3: local X;
- bits 4–7: local Y;
- bits 8–11: multi-hot action mask;
- bits 12–17: distance bucket.

The model does not use this word as a 262,144-entry categorical vocabulary.
Fields are unpacked and embedded independently. Role, stable slot, validity and
track reset are metadata embeddings derived from the sample structure.

## Policy output

The `[ACT]` hidden state is classified into five logits in strict POGEMA order:

1. `0 WAIT`
2. `1 UP`
3. `2 DOWN`
4. `3 LEFT`
5. `4 RIGHT`
