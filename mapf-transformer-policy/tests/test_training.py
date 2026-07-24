import pytest
import json
import random

import numpy as np
import torch

from mapf_transformer.training import (
    append_metric,
    capture_training_state,
    local_accumulation_steps,
    restore_training_state,
)


def test_global_accumulation_preserves_effective_batch_across_gpu_counts():
    assert local_accumulation_steps(16, 1) == 16
    assert local_accumulation_steps(16, 2) == 8


def test_global_accumulation_must_be_divisible_by_world_size():
    with pytest.raises(ValueError, match="divisible"):
        local_accumulation_steps(15, 2)


def test_append_metric_writes_json_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_metric(path, "train", step=20, loss=1.25)
    record = json.loads(path.read_text())
    assert record["event"] == "train"
    assert record["step"] == 20
    assert record["loss"] == 1.25
    assert isinstance(record["time"], float)


def test_resume_state_restores_batch_scaler_and_rng():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = capture_training_state(scaler, batch_in_epoch=123)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.random()
    np.random.rand()
    torch.rand(())
    restored_batch = restore_training_state(state, scaler)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert restored_batch == 123
    assert actual == pytest.approx(expected)
