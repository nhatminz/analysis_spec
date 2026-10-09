"""Execution-keyed proposal profiles. Host-only dispatch; CUDA queried at setup only."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DTYPES={'bf16':'torch.bfloat16','fp16':'torch.float16','fp32':'torch.float32',
        'bfloat16':'torch.bfloat16','float16':'torch.float16','float32':'torch.float32'}


def canonical_dtype(value):
    value=str(value)
    result=DTYPES.get(value,value)
    if result not in DTYPES.values():raise ValueError(f'unsupported proposal dtype: {value}')
    return result


def fingerprint(device=None):
    import torch
    import triton
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for actual OPD execution fingerprint/tuning; no fabricated profile')
    digest=hashlib.sha256()
    # Merge/normalization also affects proposal results and timings.
    for name in ('opd_reflex_kernels.py','tree_kernels.py'):
        digest.update(name.encode());digest.update((ROOT/'helper'/name).read_bytes())
    return dict(gpu=torch.cuda.get_device_name(device),compute_capability=list(torch.cuda.get_device_capability(device)),
                torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,kernel_sha256=digest.hexdigest())


def execution_key(hardware,vocab,rank,dtype,topk=16):
    return dict(**hardware,vocab=int(vocab),rank=int(rank),dtype=canonical_dtype(dtype),topk=min(int(topk),int(vocab)))


def profile_filename(key):
    gpu=re.sub(r'[^A-Za-z0-9_-]+','_',key['gpu']).strip('_')
    cc=''.join(str(x) for x in key['compute_capability'])
    dt=key['dtype'].split('.')[-1]
    # Compiler/CUDA changes must coexist rather than overwrite each other.
    tag=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:12]
    return f'{gpu}__cc{cc}__V{key["vocab"]}__r{key["rank"]}__{dt}__k{key["topk"]}__{key["kernel_sha256"][:12]}__{tag}.json'


def validate_profile(payload,key):
    actual=payload.get('execution_key')
    if actual is None:raise ValueError('legacy profile lacks execution key; retune')
    wrong=[name for name,value in key.items() if actual.get(name)!=value]
    if wrong:raise ValueError('incompatible proposal profile: '+', '.join(wrong))
    return ProposalProfile(payload)


def discover_profile(directory,key,explicit=''):
    directory=Path(directory)
    if explicit:
        path=Path(explicit)
        payload=json.loads(path.read_text());validate_profile(payload,key)
        return path,payload
    expected=directory/profile_filename(key)
    if expected.is_file():
        try:
            payload=json.loads(expected.read_text());validate_profile(payload,key)
            return expected,payload
        except (ValueError,KeyError,TypeError):pass
    # Also support manually moved/renamed profiles; never choose by model name.
    for path in sorted(directory.glob('*.json')):
        if path==expected:continue
        try:
            payload=json.loads(path.read_text());validate_profile(payload,key)
        except (ValueError,KeyError,TypeError,OSError):continue
        return path,payload
    return None,None


def _between(points,value,log=False):
    if value<=points[0]:return points[0],points[0],0.
    if value>=points[-1]:return points[-1],points[-1],0.
    for lo,hi in zip(points,points[1:]):
        if value<=hi:
            w=math.log(value/lo)/math.log(hi/lo) if log else (value-lo)/(hi-lo)
            return lo,hi,w


class ProposalProfile:
    """Interpolate measured costs in log(contexts) and linear active rows.

    Each trial measures all three backends. No assumed single crossover: GEMM
    may win at one active count and fused at another in the same context bucket.
    No Torch call, device scalar read, file I/O or launch is used by choose().
    """
    def __init__(self,payload):
        self.buckets={}
        for record in payload.get('records',[]):
            context=int(record['contexts']);trials=record['trials']
            costs={int(t['slots']):tuple(float(t[name]) for name in ('sparse','fused','gemm')) for t in trials}
            if context<1 or not costs or any(s<0 or any(not math.isfinite(v) or v<=0 for v in c) for s,c in costs.items()):
                raise ValueError('invalid measured proposal costs')
            if context in self.buckets:raise ValueError('duplicate context bucket')
            self.buckets[context]=(sorted(costs),costs)
        if not self.buckets:raise ValueError('empty measured proposal profile')
        self.contexts=sorted(self.buckets);self._cache={}

    def costs(self,contexts,active_rows):
        lo,hi,w=_between(self.contexts,max(1,int(contexts)),log=True)
        def active_cost(n):
            slots,costs=self.buckets[n]
            a,b,t=_between(slots,max(0,int(active_rows)))
            return tuple(x+(y-x)*t for x,y in zip(costs[a],costs[b]))
        return tuple(x+(y-x)*w for x,y in zip(active_cost(lo),active_cost(hi)))

    def choose(self,contexts,active_rows):
        key=(int(contexts),int(active_rows))
        result=self._cache.get(key)
        if result is None:
            costs=self.costs(*key)
            result=('sparse','fused','gemm')[min(range(3),key=costs.__getitem__)]
            if len(self._cache)>=4096:self._cache.clear()
            self._cache[key]=result
        return result


def context_shapes(batch,responses,max_draft_k,points=7):
    maximum=max(1,int(batch)*int(responses)*int(max_draft_k))
    # Geometric actual workload coverage, not absolute model-specific buckets.
    return [(n,1) for n in sorted({1,maximum,*[max(1,round(maximum**(i/max(1,points-1)))) for i in range(points)]})]


def active_trials(vocab,topk=16,points=8):
    return sorted({0,int(vocab),min(int(vocab),int(topk)),
                   *[min(int(vocab),max(1,round(vocab**(i/max(1,points-2))))) for i in range(points-1)]})


def inspect_draft(config_path,checkpoint_path,*,rank=None,dtype=None,topk=16):
    """Inspect original FastGRPO weights and FULL target config, without loading a model."""
    import torch
    config_path,checkpoint_path=map(Path,(config_path,checkpoint_path))
    config=json.loads(config_path.read_text())
    if checkpoint_path.is_dir():checkpoint_path=checkpoint_path/'draft.pth'
    payload=torch.load(checkpoint_path,map_location='meta',weights_only=True,mmap=True)
    if 'draft_model' not in payload:raise ValueError('not a FastGRPO draft checkpoint')
    state=payload['draft_model'];hidden=int(config['hidden_size']);v=int(config['vocab_size'])
    if state['fs.up_proj.weight'].shape[1]!=hidden:raise ValueError('target/draft hidden size mismatch')
    if state['states_last_norm.weight'].shape!=(hidden,):raise ValueError('invalid FastGRPO states head')
    projector=state.get('opd_projector')
    detected_rank=8 if projector is None else projector.shape[-1]
    actual_rank=detected_rank if rank is None else int(rank)
    if projector is not None and tuple(projector.shape)!=(hidden,actual_rank):raise ValueError('projector rank/hidden mismatch')
    if not 1<=actual_rank<=64:raise ValueError('OPD rank must be in [1,64]')
    checkpoint_dtype=canonical_dtype(state['fs.up_proj.weight'].dtype)
    return dict(vocab=v,rank=actual_rank,dtype=canonical_dtype(dtype or checkpoint_dtype),hidden=hidden,
                topk=min(int(topk),v),checkpoint_dtype=checkpoint_dtype,detected_rank=detected_rank,
                target_config=str(config_path),draft_checkpoint=str(checkpoint_path))
