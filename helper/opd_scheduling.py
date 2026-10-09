"""OPD-only metadata/KV optimizations. Historical baseline never imports these."""
import torch


def schedule(path,past,eos,workspace,packet,kernels=None,opd=None):
    b,capacity=path.tokens.shape
    if kernels is not None:
        if opd is not None:opd.host_sync_count+=1
        tokens,indices,mask=[x[:b,:capacity] for x in workspace[:3]];last=workspace[3][:b,:1]
        out_packet=packet[:b,:capacity+3]
        # Synchronous updates are ordered before this kernel: count is exact.
        # Async retains the pre-update snapshot to preserve overlap, never races
        # a concurrent active_count mutation or adds a wait for dispatch alone.
        reflex=opd is not None and getattr(opd,'enabled',True)
        snapshot=(opd.dispatch_snapshot if getattr(opd,'async_updates',False) else opd.active_count) if reflex else None
        kernels.pad_schedule(path,past,eos,tokens,indices,mask,last,out_packet,snapshot=snapshot)
        if reflex:
            rows=packet[:b,:capacity+4].cpu().tolist()
            opd.host_active_count=rows[0][-1]
            rows=[row[:-1] for row in rows]
        else:
            rows=out_packet.cpu().tolist()  # ONE existing HF scheduling boundary
        width=max(row[0] for row in rows)
        return rows,tokens[:,:width],indices[:,:width],mask[:,:width],last,width
    lengths=path.lengths.tolist();width=max(lengths)
    tokens,indices,mask,last=path.padded_gpu(past,width,eos,workspace=workspace)
    row_extensions=(indices==torch.arange(width)[None,:]+past).long().cumprod(1).sum(1).tolist()
    rows=[]
    for i,length in enumerate(lengths):
        rows.append([length,int(tokens[i,last[i,0]]==eos),row_extensions[i],
                     *torch.where(mask[i],indices[i],-1).tolist()])
    return rows,tokens,indices,mask,last,width


def compact_suffix_inplace(key,value,chosen,past,extension,model):
    """Gather only accepted non-prefix suffix, write back without copying past.

    OPDStaticCache appends directly into capacity storage on its next update.
    The visible prefix remains a strided view of that storage; prefix untouched.
    Scratch is capacity reused, bounded by max accepted path, NOT history size.
    """
    width=chosen.shape[1];start=past+extension;end=past+width
    if extension==width:return key[...,:end,:],value[...,:end,:]
    gather=chosen[:,extension:]
    b,heads,_,d=key.shape;n=width-extension
    index=gather[:,None,:,None].expand(b,heads,n,d)
    caches=getattr(model,'_opd_kv_scratch',None)
    if caches is None:caches={};model._opd_kv_scratch=caches
    cache=caches.get((heads,d,str(key.device),key.dtype))
    shape=(b,heads,n,d)
    if cache is None or cache[0].numel()<b*heads*n*d or cache[0].dtype!=key.dtype or cache[0].device!=key.device:
        reserve=(getattr(model,'_opd_initial_batch',b),heads,getattr(model,'_opd_max_path_capacity',width),d)
        cache=(torch.empty(reserve,device=key.device,dtype=key.dtype),torch.empty(reserve,device=value.device,dtype=value.dtype))
        caches[(heads,d,str(key.device),key.dtype)]=cache
    k=cache[0].flatten()[:b*heads*n*d].view(shape);v=cache[1].flatten()[:b*heads*n*d].view(shape)
    torch.gather(key,2,index,out=k);torch.gather(value,2,index,out=v)
    key[...,start:end,:].copy_(k);value[...,start:end,:].copy_(v)
    return key[...,:end,:],value[...,:end,:]
