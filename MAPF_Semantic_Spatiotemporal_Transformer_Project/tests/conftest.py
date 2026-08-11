import pytest

from mapf_sst.config import ModelConfig, TrainingConfig


@pytest.fixture()
def cfg() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_heads=4,
        map_layers=1,
        transformer_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )


@pytest.fixture()
def train_cfg() -> TrainingConfig:
    return TrainingConfig(
        batch_size=2,
        epochs=1,
        amp=False,
        map_loss_weight=0.02,
        semantic_reconstruction_weight=0.05,
    )
