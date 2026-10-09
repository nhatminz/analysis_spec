"""Capacity-reused OPD masks and positions; allocation only at geometric growth."""
import torch


class AttentionWorkspace:
    def __init__(self,device):
        self.device=torch.device(device)
        self.buffers={}
        self.allocations=0

    def buffer(self,name,shape,dtype):
        count=1
        for size in shape:count*=size
        key=(name,dtype);pool=self.buffers.get(key)
        if pool is None or pool.numel()<count:
            capacity=1<<max(8,(max(1,count)-1).bit_length())
            pool=torch.empty(capacity,device=self.device,dtype=dtype)
            self.buffers[key]=pool;self.allocations+=1
        return pool[:count].view(shape)

    def positions(self,name,past,width):
        positions=self.buffer(name,(width,),torch.long)
        ramp=self.buffers.get(('ramp',torch.long))
        needed=past+width
        if ramp is None or ramp.numel()<needed:
            ramp=torch.arange(1<<max(8,(max(1,needed)-1).bit_length()),device=self.device)
            self.buffers[('ramp',torch.long)]=ramp;self.allocations+=1
        positions.copy_(ramp[past:past+width])
        return positions

    def causal(self,name,past,query,batch,dtype,padding=None):
        columns=past+query
        out=self.buffer(name,(batch,1,query,columns),dtype)
        minimum=torch.finfo(dtype).min
        if self.device.type=='cuda':
            from helper.opd_attention_kernels import causal_mask
            causal_mask(out,past,padding)
        else:
            row=torch.arange(query,device=self.device)[:,None]
            col=torch.arange(columns,device=self.device)[None,:]
            out.copy_(torch.where(col<=past+row,0.,minimum).to(dtype))
            if padding is not None:out.masked_fill_(padding[:,None,None,:columns],minimum)
        return out

    def memory_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.buffers.values())
