# Validation performed

생성 환경에서 다음 검증을 완료하였다.

```text
Python 3.13.5
PyTorch 2.10.0 CPU
pytest: 7 passed
```

검증 항목:

1. 전체 context가 정확히 256 token인지 확인
2. `17×17 → 25 map tokens → 15×15 reconstruction` shape 확인
3. `25×8 = 200` structured agent token 및 `25×5` logits 확인
4. forward/backward 및 Scene encoder gradient 확인
5. Scene statistics 변경 시 Scene token 변경 확인
6. CS-PIBT-style resolver의 vertex conflict 및 edge-swap 제거 확인
7. baseline-style NPZ episode adapter 확인
8. Tiny config 2-step training, validation, checkpoint save 확인
9. 저장 checkpoint를 이용한 synthetic inference 및 resolver 실행 확인

재현 명령:

```bash
pytest -q
python train.py --config configs/tiny.yaml --max-steps 2
python inference.py --checkpoint runs/tiny/best.pt --seed 7
```
