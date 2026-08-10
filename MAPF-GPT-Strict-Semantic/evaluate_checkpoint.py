import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'MAPF_Preference_Coordination_Transformer_Project/src')]
import torch
from torch.utils.data import DataLoader
from mapf_gpt_strict import GPTConfig,MAPFGPT
from mapf_gpt_strict.expert_dataset import FastExpertDataset,collate_expert

p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/continuous_tokens_ddp_full/best.pt');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
device='cuda:0';payload=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);model=MAPFGPT(GPTConfig(**payload['config'])).to(device);model.load_state_dict(payload['model']);model.eval()
manifests=[ROOT/'data/same_expert/eval_manifest.jsonl']+sorted((ROOT/'data/same_expert').glob('eval_*_n*_manifest.jsonl'))
results={}
for manifest in manifests:
 ds=FastExpertDataset(manifest);loader=DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate_expert,pin_memory=True)
 loss_sum=correct=count=0;conf=torch.zeros(5,5,dtype=torch.long)
 with torch.no_grad():
  for batch in loader:
   batch=batch.to(device);target=batch.action;obs=batch.observation
   with torch.autocast('cuda',dtype=torch.bfloat16):logits,loss=model(obs,target,halo_maps=batch.halo_map)
   pred=logits.argmax(1);loss_sum+=float(loss)*target.numel();correct+=int((pred==target).sum());count+=target.numel()
   conf+=torch.bincount((target*5+pred).cpu(),minlength=25).reshape(5,5)
 name='overall' if manifest.name=='eval_manifest.jsonl' else manifest.stem.removeprefix('eval_').removesuffix('_manifest')
 results[name]={'samples':count,'loss':loss_sum/count,'accuracy':correct/count,'confusion':conf.tolist()};print(name,results[name])
out=ROOT/'runs/continuous_tokens_ddp_full/eval_results.json';out.write_text(json.dumps(results,indent=2));print(out)
