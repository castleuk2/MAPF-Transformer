# Integration into the MAPF Transformer Policy

## Replace the existing map encoder

The previous cell-to-latent map module can be replaced by `StructuredMapTransformer`.

```python
from mapf_map_transformer import ModelConfig, StructuredMapTransformer

self.map_encoder = StructuredMapTransformer(
    ModelConfig(
        d_model=256,
        patch_hidden_dim=256,
        num_heads=8,
        num_layers=1,
    )
)
```

The input must be a static occupancy tensor with shape `[B,17,17]`. The output used by the policy is:

```python
map_tokens = self.map_encoder(halo_maps).latent_tokens  # [B,25,256]
```

## Runtime state

Each Ego Agent keeps:

- current 17×17 static occupancy window;
- cached 25×256 map tokens;
- cached reconstruction only when diagnostics are needed;
- previous global position.

Use actual displacement rather than commanded action. A blocked move and WAIT both reuse the cached map tokens.

```python
moved = tuple(current_global_xy) != tuple(previous_global_xy)
```

When moved, derive the cardinal action from the displacement and insert the corresponding 17-cell strip. `VectorizedMapTokenRuntime` batches only the changed agents.

## Agent–Map fusion

The recommended direction is:

```text
Agent tokens = Query
Map tokens   = Key / Value
Output       = map-aware Agent tokens
```

This keeps the static geometry representation independent of transient Agent state.

The included `AgentMapCrossAttention` accepts per-Agent local coordinates and adds a relative Agent-to-patch bias.

## Reconstruction auxiliary loss

During policy training:

```python
map_output = self.map_encoder(halo_maps)
policy_loss = action_criterion(action_logits, labels)
map_loss = reconstruction_criterion(map_output, halo_maps).total
loss = policy_loss + lambda_map * map_loss
```

Start with a small coefficient such as `lambda_map=0.05–0.2` and tune using rollout performance. Reconstruction should remain a regularizer rather than dominate action prediction.

## Static/dynamic separation

Do not mark other Agents as occupied in this static map tensor. Dynamic Agent occupancy remains in Agent tokens. If dynamic obstacles must be spatially encoded, provide a separate channel/module and explicitly invalidate the latent cache when that channel changes.
