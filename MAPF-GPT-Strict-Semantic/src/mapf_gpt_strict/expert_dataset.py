from __future__ import annotations

import json, math
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .continuous_tokenizer import EncodedObservation, StableSlotAllocator

DELTAS=np.asarray(((0,0),(-1,0),(1,0),(0,-1),(0,1)),dtype=np.int64)
INF=np.iinfo(np.int32).max

@dataclass
class ExpertSample:
    observation: EncodedObservation
    halo_map: torch.Tensor
    action: torch.Tensor

    def to(self,device): return ExpertSample(self.observation.to(device),self.halo_map.to(device),self.action.to(device))

def collate_expert(samples):
    fields=[]
    for name in EncodedObservation.__dataclass_fields__:
        fields.append(torch.stack([getattr(s.observation,name) for s in samples]))
    return ExpertSample(EncodedObservation(*fields),torch.stack([s.halo_map for s in samples]),torch.stack([s.action for s in samples]))

class FastExpertDataset(Dataset):
    """Exact baseline episode/Ego/time indexing, computing only this model's features."""
    def __init__(self,manifest,goal_wait_keep_ratio=.2,max_samples=None,cache_size=4):
        self.records=[];counts=[];self.ratio=goal_wait_keep_ratio;self.cache_size=cache_size;self.cache=OrderedDict()
        mp=Path(manifest)
        for line in mp.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);arr=np.asarray(r['arrival_steps'],dtype=np.int64);t=int(r['time_steps'])
            suffix=np.maximum(0,t-arr);kept=np.where(suffix>0,np.maximum(1,np.ceil(suffix*self.ratio)).astype(np.int64),0)
            sc=arr+kept;self.records.append((Path(r['path']),arr,t,sc));counts.append(int(sc.sum()))
        self.cumulative=np.cumsum(counts);total=int(self.cumulative[-1]);self.length=min(total,max_samples) if max_samples else total

    def __len__(self):return self.length

    def _waits(self,a,t):
        suffix=max(0,t-a);keep=min(suffix,max(1,int(math.ceil(suffix*self.ratio)))) if suffix else 0
        return a+np.linspace(0,suffix-1,num=keep,dtype=np.int64) if keep else np.empty(0,dtype=np.int64)

    def _index(self,index):
        e=int(np.searchsorted(self.cumulative,index,side='right'));prev=int(self.cumulative[e-1]) if e else 0;local=index-prev
        path,arr,t,sc=self.records[e];ac=np.cumsum(sc);ego=int(np.searchsorted(ac,local,side='right'));ap=int(ac[ego-1]) if ego else 0;s=int(local-ap)
        step=s if s<int(arr[ego]) else int(self._waits(int(arr[ego]),t)[s-int(arr[ego])]);return path,step,ego

    @staticmethod
    def _stable_slot_ids(episode,ego,time_step):
        allocator=episode['slot_allocators'].setdefault(ego,StableSlotAllocator())
        plans=episode['slot_plans'].setdefault(ego,[]);positions=episode['pos']
        while len(plans)<=time_step:
            current=positions[len(plans)];ego_position=current[ego]
            visible=[i for i in range(len(current)) if np.abs(current[i]-ego_position).max()<=8]
            distance={i:int(np.abs(current[i]-ego_position).sum()) for i in visible}
            plans.append(allocator.assign_ids(ego,visible,distance))
        return plans[time_step]

    def _episode(self,path):
        p=path.resolve();e=self.cache.get(p)
        if e is None:
            with np.load(p,allow_pickle=False) as z:e={'obs':np.asarray(z['obstacles'],dtype=np.uint8),'pos':np.asarray(z['positions'],dtype=np.int64),'goals':np.asarray(z['goals'],dtype=np.int64),'act':np.asarray(z['actions'],dtype=np.int64),'dist':{},'slot_allocators':{},'slot_plans':{}}
            self.cache[p]=e
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        else:self.cache.move_to_end(p)
        return e

    @staticmethod
    def _dist(obs,goal):
        h,w=obs.shape;d=np.full((h,w),INF,dtype=np.int32);g=tuple(map(int,goal))
        if not(0<=g[0]<h and 0<=g[1]<w) or obs[g]:return d
        q=deque([g]);d[g]=0
        while q:
            r,c=q.popleft();nd=int(d[r,c])+1
            for dr,dc in DELTAS[1:]:
                x,y=r+dr,c+dc
                if 0<=x<h and 0<=y<w and not obs[x,y] and d[x,y]==INF:d[x,y]=nd;q.append((x,y))
        return d

    def __getitem__(self,index):
        path,t,ego=self._index(index);e=self._episode(path);obs,pos,goals,acts=e['obs'],e['pos'],e['goals'],e['act'];cur=pos[t];g=goals[t] if goals.ndim==3 else goals;ep=cur[ego]
        slot_ids=self._stable_slot_ids(e,ego,t)
        visible={i for i in range(len(cur)) if np.abs(cur[i]-ep).max()<=8}
        slots=[gid if gid in visible else None for gid in slot_ids]
        feat=np.zeros((256,16),np.float32);agent=np.full(256,13,np.int64);field=np.full(256,10,np.int64);lag=np.full(256,6,np.int64);role=np.zeros(256,np.int64);valid=np.zeros(256,np.int64);field[:25]=0;valid[:25]=1
        h,w=obs.shape;halo=np.ones((17,17),np.int64);r0,c0=ep-8;rs=max(0,r0);re=min(h,r0+17);cs=max(0,c0);ce=min(w,c0+17);halo[rs-r0:re-r0,cs-c0:ce-c0]=obs[rs:re,cs:ce]
        dmaps=[None]*13
        for slot,gid in enumerate(slots):
            if gid is None:continue
            goal=tuple(map(int,g[gid]));dm=e['dist'].get(goal)
            if dm is None:dm=self._dist(obs,goal);e['dist'][goal]=dm
            dmaps[slot]=dm;p=25+slot*8;rel=cur[gid]-ep;gd=g[gid]-cur[gid];dh=int(dm[tuple(cur[gid])])
            feat[p,:2]=np.clip(rel,-8,8)/8;feat[p+1,:2]=np.clip(gd,-8,8)/8;feat[p+2,:2]=(0,1) if dh==INF else (min(dh,32)/32,0)
            occupied={tuple(x) for j,x in enumerate(cur) if j!=gid}
            for a,(dr,dc) in enumerate(DELTAS):
                q=p+3+a;target=tuple(cur[gid]+(dr,dc));inside=0<=target[0]<h and 0<=target[1]<w;free=inside and not obs[target];occ=target in occupied;td=int(dm[target]) if free else INF
                delta=-1 if free and dh!=INF and td<dh else 1 if free and dh!=INF and td>dh else 0;state=2 if occ else 0 if free else 1
                feat[q,:5]=(float(free and not occ),delta,float(state==0),float(state==1),float(state==2))
            agent[p:p+8]=slot;field[p:p+8]=np.arange(1,9);lag[p:p+8]=0;role[p:p+8]=1 if slot==0 else 2;valid[p:p+8]=1
        for slot,gid in enumerate(slot_ids[:6]):
            if gid is None:continue
            goal=tuple(map(int,g[gid]));dm=e['dist'].get(goal)
            if dm is None:dm=self._dist(obs,goal);e['dist'][goal]=dm
            for step,ts in enumerate(range(t-5,t)):
                if ts<0:continue
                p=129+(slot*5+step)*4;rel=pos[ts,gid]-ep;gd=g[gid]-pos[ts,gid];dh=int(dm[tuple(pos[ts,gid])]);selected=int(acts[ts,gid]);move=pos[ts+1,gid]-pos[ts,gid];observed=next((a for a,d in enumerate(DELTAS) if np.array_equal(move,d)),0)
                feat[p,:2]=np.clip(rel,-8,8)/8;feat[p+1,:2]=np.clip(gd,-8,8)/8;feat[p+2,:2]=(0,1) if dh==INF else (min(dh,32)/32,0);feat[p+3,selected if 0<=selected<5 else 5]=1;feat[p+3,6+(observed if 0<=observed<5 else 5)]=1
                agent[p:p+4]=slot;field[p:p+4]=(1,2,3,9);lag[p:p+4]=5-step;role[p:p+4]=1 if slot==0 else 2;valid[p:p+4]=1
        enc=EncodedObservation(torch.from_numpy(feat),*[torch.from_numpy(x) for x in (agent,field,lag,role,valid)])
        return ExpertSample(enc,torch.from_numpy(halo),torch.tensor(int(acts[t,ego]),dtype=torch.long))
