from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass(slots=True)
class PolicyBatch:
    """Inputs for one or more ego-centered policy views.

    Shapes use B=batch, N=14 current slots, H=7 history tracks,
    T=4 past steps and A=5 actions.
    """

    local_maps: torch.Tensor                         # [B,17,17]

    current_xy: torch.Tensor                         # [B,N,2], signed ego-relative
    current_goal_delta: torch.Tensor                 # [B,N,2]
    current_hops: torch.Tensor                       # [B,N]
    current_valid: torch.Tensor                      # [B,N]
    current_track_reset: torch.Tensor                # [B,N]
    current_global_ids: torch.Tensor                 # [B,N], padding=-1

    candidate_target_core_xy: torch.Tensor           # [B,N,A,2], current core coordinates
    candidate_in_core: torch.Tensor                  # [B,N,A]
    candidate_static_free: torch.Tensor              # [B,N,A]
    candidate_one_hop_hops: torch.Tensor             # [B,N,A]
    candidate_delta_ctg: torch.Tensor                # [B,N,A]
    candidate_greedy: torch.Tensor                   # [B,N,A]
    candidate_bottleneck: torch.Tensor               # [B,N,A]

    history_xy: torch.Tensor                         # [B,H,T,2], current-ego frame
    history_goal_delta: torch.Tensor                 # [B,H,T,2]
    history_hops: torch.Tensor                       # [B,H,T]
    history_selected_action: torch.Tensor            # [B,H,T]
    history_observed_move: torch.Tensor              # [B,H,T]
    history_valid: torch.Tensor                      # [B,H,T]
    history_track_current_slot: torch.Tensor         # [B,H], -1 if no current slot
    history_global_ids: torch.Tensor                 # [B,H], padding=-1

    ego_action: torch.Tensor                         # [B]
    action_soft_targets: torch.Tensor | None = None  # [B,5]

    def to(self, device: torch.device | str) -> "PolicyBatch":
        values: dict[str, torch.Tensor | None] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return PolicyBatch(**values)

    @property
    def batch_size(self) -> int:
        return int(self.local_maps.shape[0])


@dataclass(slots=True)
class CommunicationGraph:
    """Mapping among ego views in a grouped scene.

    neighbor_view_index[v,k] points to the flattened ego view whose self message
    is delivered to view v. Invalid entries may be zero but must be masked.
    """

    neighbor_view_index: torch.Tensor       # [V,6]
    neighbor_valid: torch.Tensor            # [V,6]
    neighbor_current_slot: torch.Tensor     # [V,6], sender's slot in receiver view

    def to(self, device: torch.device | str) -> "CommunicationGraph":
        return CommunicationGraph(
            neighbor_view_index=self.neighbor_view_index.to(device),
            neighbor_valid=self.neighbor_valid.to(device),
            neighbor_current_slot=self.neighbor_current_slot.to(device),
        )


@dataclass(slots=True)
class SemanticReconstruction:
    current_position_x: torch.Tensor | None = None
    current_position_y: torch.Tensor | None = None
    current_goal_x: torch.Tensor | None = None
    current_goal_y: torch.Tensor | None = None
    current_hops: torch.Tensor | None = None
    history_position_x: torch.Tensor | None = None
    history_position_y: torch.Tensor | None = None
    history_goal_x: torch.Tensor | None = None
    history_goal_y: torch.Tensor | None = None
    history_hops: torch.Tensor | None = None
    history_selected_action: torch.Tensor | None = None
    history_observed_move: torch.Tensor | None = None


@dataclass(slots=True)
class PolicyOutput:
    ego_logits: torch.Tensor                      # [B,5]
    all_current_logits: torch.Tensor              # [B,14,5]
    self_message: torch.Tensor | None             # [B,D]
    map_reconstruction_logits: torch.Tensor | None
    semantic_reconstruction: SemanticReconstruction | None
    token_padding_mask: torch.Tensor              # [B,256]
    final_tokens: torch.Tensor | None = None


@dataclass(slots=True)
class PreparedPolicyContext:
    """Round-invariant tensors computed once for multi-round communication."""

    batch: PolicyBatch
    static_tokens: torch.Tensor                  # [B,249,D], field-encoded
    static_padding: torch.Tensor                 # [B,249]
    attention_bias: torch.Tensor | None           # [B,H,256,256]
    map_reconstruction_logits: torch.Tensor | None


@dataclass(slots=True)
class CommunicationOutput:
    first_pass: PolicyOutput
    final: PolicyOutput
    rounds: int


def stack_policy_batches(samples: list[PolicyBatch]) -> PolicyBatch:
    if not samples:
        raise ValueError("samples must not be empty")
    values: dict[str, torch.Tensor | None] = {}
    for field in fields(PolicyBatch):
        items = [getattr(sample, field.name) for sample in samples]
        if items[0] is None:
            if any(item is not None for item in items):
                raise ValueError(f"mixed None/non-None field: {field.name}")
            values[field.name] = None
        else:
            values[field.name] = torch.stack(items, dim=0)
    return PolicyBatch(**values)


def concatenate_policy_batches(batches: list[PolicyBatch]) -> PolicyBatch:
    """Concatenate already-batched ego views from multiple MAPF frames."""
    if not batches:
        raise ValueError("batches must not be empty")
    values: dict[str, torch.Tensor | None] = {}
    for field in fields(PolicyBatch):
        items = [getattr(batch, field.name) for batch in batches]
        if items[0] is None:
            if any(item is not None for item in items):
                raise ValueError(f"mixed None/non-None field: {field.name}")
            values[field.name] = None
        else:
            values[field.name] = torch.cat(items, dim=0)
    return PolicyBatch(**values)


def concatenate_communication_graphs(
    graphs: list[CommunicationGraph], view_counts: list[int]
) -> CommunicationGraph:
    """Concatenate frame-local graphs and offset their flattened view indices."""
    if len(graphs) != len(view_counts) or not graphs:
        raise ValueError("graphs and view_counts must have the same non-zero length")
    indices = []
    offset = 0
    for graph, count in zip(graphs, view_counts):
        indices.append(graph.neighbor_view_index + offset)
        offset += count
    return CommunicationGraph(
        neighbor_view_index=torch.cat(indices, dim=0),
        neighbor_valid=torch.cat([graph.neighbor_valid for graph in graphs], dim=0),
        neighbor_current_slot=torch.cat(
            [graph.neighbor_current_slot for graph in graphs], dim=0
        ),
    )
