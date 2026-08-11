# MPCT with Stateful C++ Feature Generator

## 새로 clone

```bash
git clone --branch mpct-cpp-feature-generator --single-branch \
  https://github.com/castleuk2/MAPF-Transformer.git
cd MAPF-Transformer/MAPF_Preference_Coordination_Transformer_CPP_Project
```

## 이미 clone한 저장소 갱신

저장소 최상위 디렉토리에서 실행한다.

```bash
git fetch origin
git switch mpct-cpp-feature-generator
git pull --ff-only origin mpct-cpp-feature-generator
cd MAPF_Preference_Coordination_Transformer_CPP_Project
```

## C++ 소스 확인 및 빌드

C++ 소스는 프로젝트 루트가 아니라 `src/mapf_pct/cpp`에 있다.

```bash
ls -lh src/mapf_pct/cpp
test -f src/mapf_pct/cpp/_feature_generator.cpp && echo "C++ source: OK"
```

```bash
python -m pip install ninja
python -m pip install -e . --no-build-isolation
python build_cpp_extension.py
```

정상 출력:

```text
C++ extension: OK (.../_feature_generator.cpython-*.so)
```

생성된 `.so`는 PC별 Python ABI가 다르므로 Git에 포함하지 않는다.

## 전체 파일 및 Dataset 검증

```bash
python verify_setup.py
pytest -q
```

정상 출력:

```text
portable setup: OK
8 passed
```

## 단일 GPU 학습

```bash
python train.py \
  --config configs/baseline_dataset_frozen_map.yaml \
  --device cuda:0 \
  --output-dir runs/mpct_portable_single_gpu
```

## GPU 2장 학습

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --config configs/baseline_dataset_frozen_map.yaml \
  --output-dir runs/mpct_portable_ddp
```

## 학습 로그 확인

```bash
tail -f runs/mpct_portable_ddp/metrics.jsonl
```

## Loss 그래프 생성

```bash
python plot_train_val_loss.py --run-dir runs/mpct_portable_ddp
```

## C++ Runtime

```python
from mapf_pct.runtime import PreferenceCoordinationPolicy

policy = PreferenceCoordinationPolicy(
    "runs/mpct_portable_ddp/best.pt",
    device="cuda:0",
    feature_backend="cpp",
)
```

## Python Runtime

```python
from mapf_pct.runtime import PreferenceCoordinationPolicy

policy = PreferenceCoordinationPolicy(
    "runs/mpct_portable_ddp/best.pt",
    device="cuda:0",
    feature_backend="python",
)
```

## Python/C++ Feature 속도 비교

```bash
PYTHONPATH=src python benchmark_feature_generators.py \
  --steps 30 \
  --output reports/cpp_feature_benchmark/result.json \
  ../pogema-mapf-transformer/data/mapf_lns2_1h/maze/val/episode_0000000_medium-mazes-eval-seed-10000_s0_n16.npz \
  ../pogema-mapf-transformer/data/mapf_lns2_1h/maze/val/episode_0000001_medium-mazes-eval-seed-10000_s0_n24.npz \
  ../pogema-mapf-transformer/data/mapf_lns2_1h/maze/val/episode_0000002_medium-mazes-eval-seed-10000_s0_n32.npz
```

## Runtime 구간별 시간 확인

```python
print(policy.last_timing)
print(policy.timing_summary())
```

```text
feature_s
batch_transfer_s
forward_s
resolver_s
total_s
```

## 주요 경로

```text
src/mapf_pct/cpp/_feature_generator.cpp       C++ 구현
src/mapf_pct/cpp/generator.py                 Python wrapper
src/mapf_pct/runtime.py                       backend 선택 및 Runtime
tests/test_cpp_feature_generator.py           Python/C++ parity
benchmark_feature_generators.py               속도 비교
checkpoints/structured_map_policy_exposure_best.pt
configs/baseline_dataset_frozen_map.yaml
```
