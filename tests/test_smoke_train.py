from __future__ import annotations

from pathlib import Path

from mapf_map_transformer.config import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig
from mapf_map_transformer.trainer import train_experiment


def test_two_step_training_smoke(tmp_path: Path) -> None:
    config = ExperimentConfig(
        model=ModelConfig(d_model=32, patch_hidden_dim=48, num_heads=4, dropout=0.0),
        dataset=DatasetConfig(train_samples=12, val_samples=6, pattern_mix=("walls", "corridors")),
        training=TrainingConfig(
            output_dir=str(tmp_path / "run"),
            epochs=1,
            batch_size=4,
            val_batch_size=3,
            max_steps=2,
            warmup_steps=0,
            num_workers=0,
            device="cpu",
            amp=False,
            log_interval=1,
            save_visualizations=0,
            include_reachability=False,
        ),
    )
    result = train_experiment(config)
    assert result.global_step == 2
    assert result.last_checkpoint.exists()
    assert result.best_checkpoint.exists()
