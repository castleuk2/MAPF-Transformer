from __future__ import annotations

from typing import Any, Mapping, Sequence

from mapf_transformer.runtime import (
    MAPFTransformerInference,
    MAPFTransformerInferenceConfig,
)


class MAPFTransformerPOGEMAPolicy:
    """Thin POGEMA policy facade around the MAPF-GPT-like inference wrapper."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        sample_actions: bool = False,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.policy = MAPFTransformerInference(
            MAPFTransformerInferenceConfig(
                checkpoint_path=checkpoint_path,
                device=device,
                sample_actions=sample_actions,
                temperature=temperature,
                seed=seed,
            )
        )

    def act(
        self,
        observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    ) -> list[int]:
        return self.policy.act(observations)

    def act_batch(self, observations_batch: Sequence[Any]) -> list[list[int]]:
        return self.policy.act_batch(observations_batch)

    def reset(self) -> None:
        self.policy.reset_states()
