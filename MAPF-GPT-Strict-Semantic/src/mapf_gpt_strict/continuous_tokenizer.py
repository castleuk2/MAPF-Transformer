from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Sequence

import torch


@dataclass(frozen=True)
class Candidate:
    valid: bool
    delta_hops: int = 0
    target_state: int = 0


@dataclass(frozen=True)
class CurrentAgent:
    position: tuple[int, int]
    relative_goal: tuple[int, int]
    remaining_hops: int
    candidates: tuple[Candidate, Candidate, Candidate, Candidate, Candidate]
    agent_id: Hashable | None = None


@dataclass(frozen=True)
class HistoryState:
    position: tuple[int, int]
    relative_goal: tuple[int, int]
    remaining_hops: int
    selected_action: int
    observed_action: int


@dataclass(frozen=True)
class EncodedObservation:
    token_features: torch.Tensor
    agent_slots: torch.Tensor
    field_ids: torch.Tensor
    lag_ids: torch.Tensor
    role_ids: torch.Tensor
    valid_ids: torch.Tensor

    def unsqueeze(self, dim: int) -> "EncodedObservation":
        return EncodedObservation(*(value.unsqueeze(dim) for value in self.__dict__.values()))

    def to(self, device) -> "EncodedObservation":
        return EncodedObservation(*(value.to(device) for value in self.__dict__.values()))


class StableSlotAllocator:
    def __init__(self, max_current=13, history_tracks=6, missed_ttl=2):
        self.max_current, self.history_tracks, self.missed_ttl = max_current, history_tracks, missed_ttl
        self.reset()

    def reset(self):
        self._slot_to_id = [None] * self.max_current
        self._missed = [0] * self.max_current

    @property
    def slot_to_id(self):
        return tuple(self._slot_to_id)

    def assign_ids(self, ego_id, visible_ids, distance_by_id):
        """Assign visible identities while retaining their prior slots.

        The method is shared by the offline dataset and online runtime so the
        exact same slot transition rule is used in both paths.
        """
        visible = set(visible_ids)
        if ego_id not in visible:
            raise ValueError("Ego must be visible")
        self._slot_to_id[0] = ego_id
        self._missed[0] = 0
        for slot in range(1, self.max_current):
            identity = self._slot_to_id[slot]
            if identity == ego_id:
                self._slot_to_id[slot], self._missed[slot] = None, 0
            elif identity in visible:
                self._missed[slot] = 0
            elif identity is not None:
                self._missed[slot] += 1
                if self._missed[slot] > self.missed_ttl:
                    self._slot_to_id[slot], self._missed[slot] = None, 0
        assigned = {identity for identity in self._slot_to_id if identity is not None}
        new_ids = [identity for identity in visible if identity not in assigned]
        new_ids.sort(key=lambda identity: (distance_by_id[identity], str(identity)))
        free = [slot for slot in range(1, self.max_current) if self._slot_to_id[slot] is None]
        for slot, identity in zip(free, new_ids):
            self._slot_to_id[slot] = identity
            self._missed[slot] = 0
        return tuple(self._slot_to_id)

    def assign(self, ego_id, agents: Sequence[CurrentAgent]):
        by_id = {a.agent_id: a for a in agents if a.agent_id is not None}
        if ego_id not in by_id:
            raise ValueError("Ego with agent_id must be present")
        ego = by_id[ego_id]
        distance = {
            identity: abs(agent.position[0] - ego.position[0]) + abs(agent.position[1] - ego.position[1])
            for identity, agent in by_id.items()
        }
        self.assign_ids(ego_id, by_id, distance)
        return [by_id.get(i) for i in self._slot_to_id]


class StableHistoryBuffer:
    def __init__(self, steps=5):
        self.steps, self._tracks = steps, {}

    def reset(self): self._tracks.clear()

    def append(self, agent_id, state):
        self._tracks.setdefault(agent_id, deque(maxlen=self.steps)).append(state)

    def tracks_for(self, slot_to_id, count=6):
        return [list(self._tracks.get(i, ())) if i is not None else [] for i in slot_to_id[:count]]


class SemanticTokenizer:
    BLOCK_SIZE, MAX_CURRENT, MAX_TRACKS, HISTORY_STEPS, FEATURE_DIM = 256, 13, 6, 5, 16

    def __init__(self, local_radius=8): self.radius = local_radius

    def _pad(self, values):
        values = list(map(float, values))
        return values + [0.0] * (self.FEATURE_DIM-len(values))

    def _xy(self, xy):
        return self._pad([max(-self.radius,min(self.radius,int(v)))/self.radius for v in xy])

    def _hops(self, h):
        return self._pad([min(h,4*self.radius)/(4*self.radius), 0.0] if h >= 0 else [0.0,1.0])

    def _candidate(self, c):
        s=max(0,min(2,int(c.target_state)))
        return self._pad([float(c.valid), max(-self.radius,min(self.radius,c.delta_hops))/self.radius, *[float(s==i) for i in range(3)]])

    def _ao(self, a, o):
        a=a if 0<=a<5 else 5; o=o if 0<=o<5 else 5
        return self._pad([float(a==i) for i in range(6)]+[float(o==i) for i in range(6)])

    def encode_stable(self, current_slots, history_by_slot):
        if len(current_slots)!=13 or current_slots[0] is None or len(history_by_slot)!=6:
            raise ValueError("expected 13 current slots (Ego first) and six matching history tracks")
        feat=[[0.0]*16 for _ in range(256)]; agent=[13]*256; field=[10]*256
        lag=[6]*256; role=[0]*256; valid=[0]*256
        field[:25]=[0]*25; valid[:25]=[1]*25
        for slot,a in enumerate(current_slots):
            if a is None: continue
            p=25+slot*8
            feat[p:p+8]=[self._xy(a.position),self._xy(a.relative_goal),self._hops(a.remaining_hops),*[self._candidate(c) for c in a.candidates]]
            agent[p:p+8]=[slot]*8; field[p:p+8]=list(range(1,9)); lag[p:p+8]=[0]*8
            role[p:p+8]=[1 if slot==0 else 2]*8; valid[p:p+8]=[1]*8
        for slot,track in enumerate(history_by_slot):
            if len(track)>5: raise ValueError("history is limited to five states")
            first=5-len(track)
            for step,s in enumerate(track,start=first):
                p=129+(slot*5+step)*4
                feat[p:p+4]=[self._xy(s.position),self._xy(s.relative_goal),self._hops(s.remaining_hops),self._ao(s.selected_action,s.observed_action)]
                agent[p:p+4]=[slot]*4; field[p:p+4]=[1,2,3,9]; lag[p:p+4]=[5-step]*4
                role[p:p+4]=[1 if slot==0 else 2]*4; valid[p:p+4]=[1]*4
        return EncodedObservation(torch.tensor(feat),*[torch.tensor(x,dtype=torch.long) for x in (agent,field,lag,role,valid)])
