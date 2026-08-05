# Architecture Specification

## Tensor shapes

| Stage | Shape |
|---|---|
| Halo input | `[B,17,17]` |
| Core map | `[B,15,15]` |
| Raw patches | `[B,25,3,3]` |
| Patch features | `[B,25,34]` in the binary default |
| Patch embedding | `[B,25,256]` |
| Structured latent tokens | `[B,25,256]` |
| Per-token reconstruction | `[B,25,9]` |
| Reconstructed core | `[B,15,15]` |

## Coordinate convention

- Array row increases downward.
- Array column increases rightward.
- Agent local `x` increases rightward.
- Agent local `y` increases upward.
- The Ego is at the central core cell `(row=7,col=7)`, which is halo index `(8,8)`.
- Patch centers in local `(x,y)` are `{-6,-3,0,3,6}` on both axes.

## Unified patch port definition

For a patch edge cell `s` and the cardinally adjacent cell `n`, the port is open exactly when both are known free cells.

\[
\mathrm{open}(s,n)=\mathbb{I}[s=0]\mathbb{I}[n=0]
\]

For internal patch edges, `n` belongs to another core patch. For the outer core edge, `n` belongs to the one-cell halo. Thus no artificial occupied boundary is introduced at the 15×15 crop boundary.

Each direction has a 3-bit opening pattern. The pattern is also encoded as an integer in `[0,7]` for connectivity attention bias.

## Attention

A single full-attention encoder block is the default. Query, Key and Value are independently projected from the same 25 input patch tokens.

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

The output remains 25 tokens because this is self-attention, not latent-query cross-attention.

The learned relative-position table has shape:

\[
[H,(2\cdot5-1)^2]=[H,81]
\]

The connectivity table has shape:

\[
[H,4,8]
\]

where the final axes represent source direction and 3-bit port code.

## Decoder and bottleneck interpretation

The shared decoder maps each final token to nine logits. Because every token has a fixed spatial anchor, the decoder does not need an output query or permutation matching.

A successful reconstruction demonstrates that each 256-dimensional patch token and the 25-token set preserve occupancy geometry. It does not alone prove that the representation is optimal for action selection. The next-stage policy experiment should compare reconstruction metrics and rollout metrics jointly.
