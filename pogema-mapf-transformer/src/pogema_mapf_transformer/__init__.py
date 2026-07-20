"""Non-invasive POGEMA integration for the MAPF Transformer policy."""

from .config import DatasetGenerationConfig, EvaluationConfig

__all__ = [
    "DatasetGenerationConfig",
    "EvaluationConfig",
    "POGEMAMAPFTransformerAdapter",
    "MAPFTransformerPOGEMAPolicy",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Keep policy-only dependencies lazy for CPU dataset-generation workers."""
    if name == "POGEMAMAPFTransformerAdapter":
        from .env_adapter import POGEMAMAPFTransformerAdapter

        return POGEMAMAPFTransformerAdapter
    if name == "MAPFTransformerPOGEMAPolicy":
        from .policy_adapter import MAPFTransformerPOGEMAPolicy

        return MAPFTransformerPOGEMAPolicy
    raise AttributeError(name)
