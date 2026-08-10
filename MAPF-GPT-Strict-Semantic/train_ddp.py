from __future__ import annotations

import argparse, json, math, os, random, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
PCT=ROOT.parent/'MAPF_Preference_Coordination_Transformer_Project'
sys.path.insert(0,str(PCT/'src'))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from mapf_gpt_strict import GPTConfig, MAPFGPT
from mapf_gpt_strict.expert_dataset import FastExpertDataset,collate_expert


def args():
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=24)
 p.add_argument('--workers',type=int,default=10);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--max-train',type=int)
 p.add_argument('--max-val',type=int,default=65536);p.add_argument('--output',default='runs/continuous_tokens_ddp')
 p.add_argument('--map-checkpoint',type=Path,default=ROOT.parent/'mapf-structured-map-transformer/runs/policy_exposure_structured_25_ce/best.pt')
 return p.parse_args()

def main():
 a=args(); rank=int(os.environ.get('RANK',0)); local=int(os.environ.get('LOCAL_RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
 if world>1: dist.init_process_group('nccl'); torch.cuda.set_device(local)
 device=torch.device(f'cuda:{local}'); torch.manual_seed(42+rank);np.random.seed(42+rank);random.seed(42+rank)
 train=FastExpertDataset(ROOT/'data/same_expert/train_manifest.jsonl',max_samples=a.max_train)
 val=FastExpertDataset(ROOT/'data/same_expert/val_manifest.jsonl',max_samples=a.max_val)
 # Sequential episode-local indexing is intentional: it preserves the exact
 # sample set while allowing each worker's NPZ/distance-map cache to be reused.
 ts=DistributedSampler(train,world,rank,shuffle=False,seed=42) if world>1 else None
 train_loader=DataLoader(train,batch_size=a.batch_size,sampler=ts,shuffle=False,num_workers=a.workers,collate_fn=collate_expert,pin_memory=True,persistent_workers=a.workers>0)
 val_loader=DataLoader(val,batch_size=a.batch_size*2,shuffle=False,num_workers=max(1,a.workers//2),collate_fn=collate_expert,pin_memory=True)
 model=MAPFGPT(GPTConfig(dropout=.1)).to(device)
 if not a.map_checkpoint.is_file():raise FileNotFoundError(f'map checkpoint not found: {a.map_checkpoint}')
 model.load_map_encoder(a.map_checkpoint,freeze=True)
 if world>1:model=DDP(model,device_ids=[local],find_unused_parameters=False)
 opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=.1,betas=(.9,.95))
 total_steps=len(train_loader)*a.epochs
 def lr_scale(step):
  if step<2000:return max(1,step)/2000
  progress=min(1.0,(step-2000)/max(1,total_steps-2000))
  return .1+.9*.5*(1+math.cos(math.pi*progress))
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_scale)
 scaler=torch.amp.GradScaler('cuda'); out=ROOT/a.output
 if rank==0:out.mkdir(parents=True,exist_ok=True)
 best=1e9; step=0
 for epoch in range(a.epochs):
  if ts:ts.set_epoch(epoch)
  model.train(); total=correct=count=0
  for batch in train_loader:
   batch=batch.to(device); obs=batch.observation; target=batch.action
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16): logits,loss=model(obs,target,halo_maps=batch.halo_map)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();scheduler.step()
   total+=float(loss.detach())*target.numel();correct+=int((logits.argmax(1)==target).sum());count+=target.numel();step+=1
   if rank==0 and step%100==0:print(f'epoch={epoch+1} step={step} train_loss={total/count:.6f} train_acc={correct/count:.6f} lr={opt.param_groups[0]["lr"]:.8f}',flush=True)
  if world>1:
   stats=torch.tensor([total,correct,count],device=device);dist.all_reduce(stats);total,correct,count=stats.tolist()
  if rank==0:
   eval_model=model.module if hasattr(model,'module') else model
   eval_model.eval();vt=vc=vn=0
   with torch.no_grad():
    for batch in val_loader:
     batch=batch.to(device);obs=batch.observation;target=batch.action
     with torch.autocast('cuda',dtype=torch.bfloat16): logits,loss=eval_model(obs,target,halo_maps=batch.halo_map)
     vt+=float(loss)*target.numel();vc+=int((logits.argmax(1)==target).sum());vn+=target.numel()
   event={'epoch':epoch+1,'step':step,'train_loss':total/count,'train_accuracy':correct/count,'val_loss':vt/vn,'val_accuracy':vc/vn}
   print(json.dumps(event),flush=True);open(out/'metrics.jsonl','a').write(json.dumps(event)+'\n')
   raw=model.module if hasattr(model,'module') else model;payload={'model':raw.state_dict(),'config':raw.cfg.__dict__,'metrics':event}
   torch.save(payload,out/'last.pt')
   if event['val_loss']<best:best=event['val_loss'];torch.save(payload,out/'best.pt')
  if world>1:dist.barrier()
 if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
