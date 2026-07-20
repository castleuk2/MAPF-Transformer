import pytest
import json

from mapf_transformer.training import append_metric, local_accumulation_steps


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
