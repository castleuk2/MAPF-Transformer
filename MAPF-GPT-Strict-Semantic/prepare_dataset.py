import argparse, json, math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Prepare portable Strict Semantic manifests')
parser.add_argument('--train-manifest',type=Path,required=True)
parser.add_argument('--val-manifest',type=Path,required=True)
parser.add_argument('--eval-manifest',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=ROOT/'data/same_expert')
args=parser.parse_args()
SOURCES={'train':args.train_manifest,'val':args.val_manifest,'eval':args.eval_manifest}
out=args.output_dir;out.mkdir(parents=True,exist_ok=True);summary={}
for split,source in SOURCES.items():
 records=[];counts=Counter();samples=0;effective_samples=0
 for line in source.read_text().splitlines():
  r=json.loads(line);raw=r.get('source_path') or r['path'];p=Path(raw)
  if not p.is_absolute():p=(source.parent/p).resolve()
  if not p.exists():raise FileNotFoundError(p)
  r['path']=str(p);r.setdefault('arrival_steps',[r['time_steps']]*r['num_agents'])
  records.append(r);counts[f"{r['map_family']}_n{r['num_agents']}"]+=1;samples+=int(r.get('num_samples',0))
  # FastExpertDataset과 동일한 Ego/time indexing: 도착 전 전 구간과
  # 도착 후 WAIT 구간의 20%를 학습 표본으로 사용한다.
  t=int(r['time_steps'])
  for arrival in r['arrival_steps']:
   suffix=max(0,t-int(arrival))
   effective_samples+=int(arrival)+(max(1,math.ceil(suffix*.2)) if suffix else 0)
 target=out/f'{split}_manifest.jsonl';target.write_text(''.join(json.dumps(r)+'\n' for r in records))
 groups={}
 for r in records: groups.setdefault(f"{r['map_family']}_n{r['num_agents']}",[]).append(r)
 for name,items in groups.items():
  (out/f'{split}_{name}_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
 summary[split]={'episodes':len(records),'source_declared_samples':samples,'effective_samples':effective_samples,'groups':dict(sorted(counts.items())),'manifest':str(target)}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
