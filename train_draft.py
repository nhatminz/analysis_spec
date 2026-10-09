"""FastGRPO ShareGPT pretraining with SpecNaacl paths and resumable checkpoints."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from helper.fastgrpo_model import FastGRPOModel
from helper.pretrain_data import DataCollator
from helper.checkpointing import capture_rng_state, restore_rng_state


def pretrain_loss(model, batch):
    """Exact slices, assistant mask, per-sequence mean, and softmax/log of upstream."""
    input_ids, attention_mask, loss_mask=(batch[k].to(model.device) for k in ('input_ids','attention_mask','loss_mask'))
    with torch.no_grad():
        outputs=model.target_model.model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=False)
        feature_states=outputs.last_hidden_state
        target_logits=model.target_model.lm_head(feature_states)[:,:-1,:]
    feature_states=feature_states[:,:-1,:].to(model.dtype)
    input_ids=input_ids[:,1:];attention_mask=attention_mask[:,:-1];loss_mask=loss_mask[:,:-1]
    outputs=model(hidden_states=feature_states,input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
    next_feature_states=outputs['next_feature_states']
    draft_logits=model.lm_head(outputs['hidden_states'].to(model.target_model.dtype))
    loss1=torch.nn.functional.smooth_l1_loss(next_feature_states[:,:-1,:].float(),feature_states[:,1:,:].float(),reduction='none')
    loss1=torch.mean(loss1,dim=-1)*loss_mask[:,:-1]
    loss1=torch.sum(loss1,dim=-1)/torch.sum(loss_mask[:,:-1],dim=-1)
    loss1=loss1.mean()*2.0
    with torch.no_grad():target_logits=target_logits[:,1:,:].float().softmax(dim=-1).detach()
    draft_logits=draft_logits[:,:-1,:].float().softmax(dim=-1)
    loss2=torch.sum(target_logits*torch.log(draft_logits),dim=-1)*loss_mask[:,:-1]
    loss2=torch.sum(loss2,dim=-1)/torch.sum(loss_mask[:,:-1],dim=-1)
    loss2=-loss2.mean()*0.1
    return loss1,loss2


def atomic_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model_dir','dataset_dir','saved_model_dir','log_dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--version_name',default='fastgrpo-pretrain')
    p.add_argument('--model_type',default='qwen2');p.add_argument('--num_epochs',type=int,default=5)
    p.add_argument('--batch_size',type=int,default=4);p.add_argument('--accumulation_steps',type=int,default=1)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--warmup_ratio',type=float,default=.05)
    p.add_argument('--max_length',type=int,default=2048);p.add_argument('--max_samples',type=int,default=0)
    p.add_argument('--num_workers',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save_interval',type=int,default=500);p.add_argument('--resume',default='')
    p.add_argument('--max_steps',type=int,default=0,help='Stop after this optimizer step, save resumable state; 0 means all epochs')
    p.add_argument('--model_output_root',default='');p.add_argument('--dtype',default='bf16',choices=['bf16','fp16','fp32','auto'])
    p.add_argument('--attn_implementation',default='sdpa')
    a=p.parse_args(argv)
    if min(a.batch_size,a.accumulation_steps,a.num_epochs,a.max_length,a.save_interval)<1: p.error('positive training settings required')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:dist.init_process_group('nccl')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    config=AutoConfig.from_pretrained(a.model_dir)
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32,'auto':'auto'}[a.dtype]
    target=AutoModelForCausalLM.from_pretrained(a.model_dir,torch_dtype=dtype,attn_implementation=a.attn_implementation).cuda().eval()
    config=deepcopy(config);config.num_hidden_layers=1;config.rope_scaling=None;config.torch_dtype=target.dtype
    model=FastGRPOModel(config,target).cuda()
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,padding_side='right')
    data=json.loads(Path(a.dataset_dir).read_text())
    if a.max_samples:data=data[:a.max_samples]
    sampler=DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=a.seed)
    # A separate generator prevents iterator construction on resume from consuming model RNG.
    loader=DataLoader(data,batch_size=a.batch_size,sampler=sampler,
        collate_fn=DataCollator(tokenizer,a.max_length,a.model_type),num_workers=a.num_workers,
        persistent_workers=a.num_workers>0,generator=torch.Generator().manual_seed(a.seed))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    total_steps=a.num_epochs*((len(loader)+a.accumulation_steps-1)//a.accumulation_steps)
    scheduler=get_scheduler('cosine_with_min_lr',optimizer=optimizer,
        num_warmup_steps=min(int(a.warmup_ratio*total_steps),500),num_training_steps=total_steps,
        scheduler_specific_kwargs={'min_lr_rate':0.})
    output=Path(a.saved_model_dir);logs=Path(a.log_dir)
    output.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    latest=output/(a.version_name+'-latest');latest.mkdir(exist_ok=True)
    step=accumulated=epoch_start=batch_start=0
    optimizer.zero_grad(set_to_none=True)
    if a.resume:
        path=latest/'training_state.pt' if a.resume=='auto' else Path(a.resume)
        if path.is_dir():path=path/'training_state.pt'
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['world_size']!=world:raise ValueError('resume world size mismatch')
            model.draft_model.load_state_dict(state['draft_model'])
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            step,accumulated,epoch_start,batch_start=(state[k] for k in ('step','accumulated','epoch','next_batch'))
            local=state['ranks'][rank]
            for name,param in model.draft_model.named_parameters():
                if name in local['gradients']:param.grad=local['gradients'][name].to(param.device)
            restore_rng_state(local['rng'])
        elif a.resume!='auto':raise FileNotFoundError(path)
    def save(epoch,next_batch):
        local=dict(rng=capture_rng_state(),gradients={n:p.grad.cpu() for n,p in model.draft_model.named_parameters() if p.grad is not None})
        ranks=[None]*world
        if world>1:dist.all_gather_object(ranks,local)
        else:ranks=[local]
        if rank:return
        weights={'draft_model':model.draft_model.state_dict()}
        atomic_save(weights,output/f'step{step}.pth');atomic_save(weights,latest/'draft.pth')
        atomic_save(dict(**weights,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            step=step,accumulated=accumulated,epoch=epoch,next_batch=next_batch,world_size=world,ranks=ranks),latest/'training_state.pt')
        # Full target config supplies vocabulary/head dimensions for the proposal tuner.
        target.config.to_json_file(latest/'target_config.json')
        if a.model_output_root:
            root=Path(a.model_output_root);root.mkdir(parents=True,exist_ok=True)
            for name,destination in [('latest_checkpoint',latest),('latest_target_config.json',latest/'target_config.json')]:
                link=root/name;temporary=root/(name+'.tmp')
                temporary.unlink(missing_ok=True);temporary.symlink_to(destination.resolve());temporary.replace(link)
    start=time.perf_counter()
    progress_disabled=rank!=0 or os.environ.get('TQDM_DISABLE','').strip().lower() in {'1','true','yes','on'}
    for epoch in range(epoch_start,a.num_epochs):
        sampler.set_epoch(epoch)
        progress=tqdm(enumerate(loader),total=len(loader),desc=f'Pretrain epoch {epoch+1}/{a.num_epochs}',
                      unit='batch',dynamic_ncols=True,disable=progress_disabled)
        for i,batch in progress:
            if epoch==epoch_start and i<batch_start:continue
            has_labels=torch.any(batch['loss_mask']==1).to(device=model.device,dtype=torch.int32)
            if world>1:dist.all_reduce(has_labels,op=dist.ReduceOp.MIN)
            if not has_labels:continue
            loss1,loss2=pretrain_loss(model,batch);loss=loss1+loss2
            valid=torch.isfinite(loss).to(torch.int32)
            if world>1:dist.all_reduce(valid,op=dist.ReduceOp.MIN)
            if not valid:continue
            accumulated+=1;(loss/a.accumulation_steps).backward()
            if accumulated%a.accumulation_steps:continue
            if world>1:
                for param in model.draft_model.parameters():
                    if param.grad is not None:dist.all_reduce(param.grad);param.grad.div_(world)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            if rank==0:
                row=dict(step=step,epoch=epoch,batch=i,loss=float(loss.detach()),loss1=float(loss1.detach()),loss2=float(loss2.detach()),wall_time_s=time.perf_counter()-start)
                progress.set_postfix(step=step,loss=f"{row['loss']:.4f}",
                    feature=f"{row['loss1']:.4f}",ce=f"{row['loss2']:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",refresh=False)
                for filename in ('metrics.jsonl',f'epoch_{epoch}.log'):
                    with (logs/filename).open('a') as f:f.write(json.dumps(row)+'\n')
            if step%a.save_interval==0:save(epoch,i+1)
            if a.max_steps>0 and step>=a.max_steps:
                if not progress_disabled:
                    progress.n=i+1
                    progress.refresh()
                progress.close()
                save(epoch,i+1)
                if world>1:dist.destroy_process_group()
                return
        progress.close()
        batch_start=0
    save(a.num_epochs,0)
    if rank==0:
        (output/'pretrain_complete.json').write_text(json.dumps(dict(step=step,epochs=a.num_epochs,target_model_path=str(Path(a.model_dir).resolve())))+'\n')
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
