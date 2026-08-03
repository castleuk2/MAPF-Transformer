# Validation Record

Validation was performed in the supplied execution environment with PyTorch 2.10 on CPU.

## Automated tests

```text
12 passed
```

Covered cases:

- 15×15 ↔ 25×3×3 patch round-trip.
- 17×17 halo control of outer patch ports.
- UP/DOWN/LEFT/RIGHT 17-cell strip updates against fresh global-map crops.
- 25×D token and 15×15 reconstruction tensor shapes.
- Forward/backward propagation through the map encoder.
- Agent-to-map cross-attention shape validation.
- Exact latent tensor reuse for WAIT.
- Vectorized runtime encoding only changed agents.
- Perfect-reconstruction metric sanity checks.
- Two-step end-to-end training/checkpoint smoke test.

## End-to-end command validation

The following were executed successfully:

```bash
python train.py --config configs/smoke.yaml --output-dir runs/smoke_validation --max-steps 3
python evaluate.py --checkpoint runs/smoke_validation/best.pt
python demo_runtime.py --checkpoint runs/smoke_validation/best.pt --device cpu
python examples/integration_snippet.py
```

The three-step checkpoint is only a software smoke test and is not evidence that 25 tokens are sufficient. Token adequacy must be judged after full training with occupied IoU, boundary F1, reachability IoU, and downstream MAPF rollout metrics.
