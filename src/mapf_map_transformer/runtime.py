from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .actions import ACTION_TO_DELTA, Action, coerce_action
from .geometry import crop_global_map, incoming_strip_from_global, shift_halo_map
from .model import StructuredMapTransformer


@dataclass(slots=True)
class RuntimeMapOutput:
    latent_tokens: torch.Tensor
    reconstruction_probabilities: torch.Tensor
    reused: bool
    version: int
    halo_map: np.ndarray


class GlobalMapWindowProvider:
    """Extract 17x17 windows and the single incoming 17-cell strip from a static map."""

    def __init__(self, global_map: np.ndarray, *, pad_value: int = 1) -> None:
        self.global_map = np.asarray(global_map)
        if self.global_map.ndim != 2:
            raise ValueError("global_map must be two-dimensional.")
        self.pad_value = int(pad_value)

    def crop(self, center_rc: tuple[int, int]) -> np.ndarray:
        return crop_global_map(self.global_map, center_rc, size=17, pad_value=self.pad_value)

    def incoming_strip(self, new_center_rc: tuple[int, int], action: Action | int | str) -> np.ndarray:
        return incoming_strip_from_global(
            self.global_map,
            new_center_rc,
            action,
            size=17,
            pad_value=self.pad_value,
        )


class MapTokenRuntime:
    """Stateful single-agent runtime with exact WAIT/failed-move latent reuse."""

    def __init__(self, model: StructuredMapTransformer, *, device: torch.device | str | None = None) -> None:
        self.model = model
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._halo_map: np.ndarray | None = None
        self._latent_tokens: torch.Tensor | None = None
        self._reconstruction: torch.Tensor | None = None
        self.version = 0
        self.encode_count = 0
        self.reuse_count = 0

    @property
    def initialized(self) -> bool:
        return self._halo_map is not None

    def reset(self, halo_map: np.ndarray | torch.Tensor) -> RuntimeMapOutput:
        array = self._normalize_halo(halo_map)
        self._halo_map = array
        self.version = 0
        self.encode_count = 0
        self.reuse_count = 0
        self._encode_current()
        return self._result(reused=False)

    def step(
        self,
        action: Action | int | str,
        *,
        moved: bool,
        incoming_strip: np.ndarray | Sequence[int] | None = None,
        full_halo_map: np.ndarray | torch.Tensor | None = None,
        static_map_changed: bool = False,
    ) -> RuntimeMapOutput:
        self._require_initialized()
        action = coerce_action(action)

        if not moved or action == Action.WAIT:
            if moved and action == Action.WAIT:
                raise ValueError("WAIT cannot correspond to a nonzero actual displacement.")
            if static_map_changed:
                if full_halo_map is None:
                    raise ValueError("A changed static map requires full_halo_map to resynchronize the cache.")
                self._halo_map = self._normalize_halo(full_halo_map)
                self.version += 1
                self._encode_current()
                return self._result(reused=False)
            self.reuse_count += 1
            return self._result(reused=True)

        if action == Action.WAIT:
            raise ValueError("A successful movement requires a cardinal action.")
        if full_halo_map is not None:
            updated = self._normalize_halo(full_halo_map)
            if incoming_strip is not None:
                shifted = shift_halo_map(self._halo_map, action, incoming_strip)
                if not np.array_equal(updated, shifted):
                    raise ValueError("full_halo_map does not match the incremental 17-cell strip update.")
        else:
            if incoming_strip is None:
                raise ValueError("A successful move requires incoming_strip or full_halo_map.")
            updated = shift_halo_map(self._halo_map, action, incoming_strip)

        self._halo_map = updated
        self.version += 1
        self._encode_current()
        return self._result(reused=False)

    def step_from_positions(
        self,
        previous_center_rc: tuple[int, int],
        new_center_rc: tuple[int, int],
        provider: GlobalMapWindowProvider,
    ) -> RuntimeMapOutput:
        delta = (new_center_rc[0] - previous_center_rc[0], new_center_rc[1] - previous_center_rc[1])
        reverse = {value: key for key, value in ACTION_TO_DELTA.items()}
        if delta not in reverse:
            raise ValueError(f"Only WAIT or one-cell cardinal displacement is supported, got delta={delta}.")
        action = reverse[delta]
        if action == Action.WAIT:
            return self.step(action, moved=False)
        strip = provider.incoming_strip(new_center_rc, action)
        return self.step(action, moved=True, incoming_strip=strip)

    def _normalize_halo(self, halo_map: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(halo_map, torch.Tensor):
            halo_map = halo_map.detach().cpu().numpy()
        array = np.asarray(halo_map, dtype=np.int64)
        if array.shape != (17, 17):
            raise ValueError(f"Expected halo map shape (17,17), got {array.shape}")
        if array.min() < 0 or array.max() >= self.model.config.num_cell_states:
            raise ValueError("halo_map contains a cell state outside the configured range.")
        return np.array(array, copy=True)

    @torch.inference_mode()
    def _encode_current(self) -> None:
        assert self._halo_map is not None
        tensor = torch.from_numpy(self._halo_map).to(device=self.device, dtype=torch.long).unsqueeze(0)
        output = self.model(tensor)
        self._latent_tokens = output.latent_tokens[0].detach()
        self._reconstruction = torch.sigmoid(output.reconstruction_logits[0]).detach()
        self.encode_count += 1

    def _result(self, *, reused: bool) -> RuntimeMapOutput:
        self._require_initialized()
        assert self._latent_tokens is not None and self._reconstruction is not None and self._halo_map is not None
        return RuntimeMapOutput(
            latent_tokens=self._latent_tokens,
            reconstruction_probabilities=self._reconstruction,
            reused=reused,
            version=self.version,
            halo_map=np.array(self._halo_map, copy=True),
        )

    def _require_initialized(self) -> None:
        if self._halo_map is None:
            raise RuntimeError("Call reset() before step().")


class VectorizedMapTokenRuntime:
    """Batch multiple agent map memories and encode only agents whose window changed."""

    def __init__(self, model: StructuredMapTransformer, *, device: torch.device | str | None = None) -> None:
        self.model = model
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.halo_maps: list[np.ndarray] = []
        self.latent_tokens: list[torch.Tensor] = []
        self.reconstructions: list[torch.Tensor] = []
        self.versions: list[int] = []
        self.encode_count = 0
        self.reuse_count = 0

    @torch.inference_mode()
    def reset(self, halo_maps: Sequence[np.ndarray | torch.Tensor]) -> list[RuntimeMapOutput]:
        normalized = [self._normalize(item) for item in halo_maps]
        if not normalized:
            raise ValueError("halo_maps must not be empty.")
        batch = torch.from_numpy(np.stack(normalized)).to(self.device, dtype=torch.long)
        output = self.model(batch)
        self.halo_maps = normalized
        self.latent_tokens = [token.detach() for token in output.latent_tokens]
        probabilities = torch.sigmoid(output.reconstruction_logits)
        self.reconstructions = [item.detach() for item in probabilities]
        self.versions = [0 for _ in normalized]
        self.encode_count = len(normalized)
        self.reuse_count = 0
        return [self._result(index, reused=False) for index in range(len(normalized))]

    @torch.inference_mode()
    def step(
        self,
        actions: Sequence[Action | int | str],
        moved: Sequence[bool],
        incoming_strips: Sequence[np.ndarray | Sequence[int] | None],
    ) -> list[RuntimeMapOutput]:
        count = len(self.halo_maps)
        if not (len(actions) == len(moved) == len(incoming_strips) == count):
            raise ValueError("actions, moved and incoming_strips must match the number of runtime states.")

        changed_indices: list[int] = []
        for index, (raw_action, did_move, strip) in enumerate(zip(actions, moved, incoming_strips, strict=True)):
            action = coerce_action(raw_action)
            if not did_move or action == Action.WAIT:
                if did_move and action == Action.WAIT:
                    raise ValueError("WAIT cannot correspond to movement.")
                self.reuse_count += 1
                continue
            if strip is None:
                raise ValueError(f"Agent {index}: successful move requires a 17-cell incoming strip.")
            self.halo_maps[index] = shift_halo_map(self.halo_maps[index], action, strip)
            self.versions[index] += 1
            changed_indices.append(index)

        if changed_indices:
            batch_np = np.stack([self.halo_maps[index] for index in changed_indices])
            batch = torch.from_numpy(batch_np).to(self.device, dtype=torch.long)
            output = self.model(batch)
            probabilities = torch.sigmoid(output.reconstruction_logits)
            for local_index, global_index in enumerate(changed_indices):
                self.latent_tokens[global_index] = output.latent_tokens[local_index].detach()
                self.reconstructions[global_index] = probabilities[local_index].detach()
            self.encode_count += len(changed_indices)

        changed_set = set(changed_indices)
        return [self._result(index, reused=index not in changed_set) for index in range(count)]

    def _normalize(self, item: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        array = np.asarray(item, dtype=np.int64)
        if array.shape != (17, 17):
            raise ValueError(f"Expected (17,17), got {array.shape}")
        return np.array(array, copy=True)

    def _result(self, index: int, *, reused: bool) -> RuntimeMapOutput:
        return RuntimeMapOutput(
            latent_tokens=self.latent_tokens[index],
            reconstruction_probabilities=self.reconstructions[index],
            reused=reused,
            version=self.versions[index],
            halo_map=np.array(self.halo_maps[index], copy=True),
        )
