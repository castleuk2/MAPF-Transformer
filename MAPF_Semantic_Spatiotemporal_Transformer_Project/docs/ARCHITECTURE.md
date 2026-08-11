# Architecture

```text
17×17 Ego-centered Map
        │
        ▼
StructuredPatchMapEncoder
        │ 25 fixed patch tokens
        ├───────────────────────────────────────────────┐
        │                                               │
Current Agent Tokenizer                          History Tokenizer
14 × [P,G,R,5 Candidates]                        7 × 4 × [P,G,R,AO]
        │                                               │
        └────── Candidate source/target spatial fusion ─┘
                              │
                              ▼
Structured Spatiotemporal Transformer
  • field-pair bias
  • relative-time bias
  • same-agent / same-track bias
  • vertex / edge-swap / occupancy / corridor bias
  • candidate↔map source/target bias
                              │
                              ├─ Optional self/neighbor messages
                              │
                              ▼
Ego five Candidate outputs → shared score head → [5] preference
```

## Encoder-only 이유

현재 문제는 현재 action을 한 번에 분류하는 문제이며 action sequence를 autoregressive하게 생성하지 않음. 미래 action이나 현재 GT는 입력하지 않으므로 causal mask가 필요하지 않음.

## Communication

각 Ego view의 local forward는 독립적임. `MultiRoundCommunicationPolicy`가 명시적인 gather/fusion 단계를 수행할 때만 서로 다른 Ego view의 hidden state가 교환됨.
