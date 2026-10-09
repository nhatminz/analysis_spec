"""OPD-only contiguous append pools; historical FastGRPO uses its own history.

Finished rows retain lengths, not tensor views. A rare capacity grow therefore
cannot pin an old allocation. Only the final materialization makes owned output
storage, once per field, after generation has ended.
"""
import torch


class ContiguousRolloutHistory:
    def __init__(self, initial, *, repeats=1, max_length, reserve_tokens=256):
        if repeats<1 or max_length<1:
            raise ValueError('positive history repeats and capacity required')
        sizes={t.shape[0] for t in initial.values()}
        if len(sizes)!=1:raise ValueError('history batch sizes differ')
        self.response_count=sizes.pop()*repeats
        self.buffers={};self.lengths={};self.capacities={};self.finished={}
        for name,tensor in initial.items():
            length=tensor.shape[1];capacity=max(length,int(max_length))
            pool=tensor.new_empty((self.response_count,capacity,*tensor.shape[2:]))
            pool[:,:length].view(tensor.shape[0],repeats,length,*tensor.shape[2:]).copy_(tensor[:,None])
            self.buffers[name]=pool;self.lengths[name]=length;self.capacities[name]=capacity

    def append(self, original_rows, chunks, owners=None):
        if set(chunks)!=set(self.buffers):raise ValueError('history fields differ')
        if owners is None:
            owners=torch.as_tensor(original_rows,device=next(iter(chunks.values())).device,dtype=torch.long)
        for name,chunk in chunks.items():
            pool=self.buffers[name]
            if chunk.shape[0]!=len(original_rows) or chunk.shape[2:]!=pool.shape[2:] or chunk.dtype!=pool.dtype:
                raise ValueError('history chunk shape/dtype changed')
            start=self.lengths[name];end=start+chunk.shape[1]
            if end>pool.shape[1]:
                # Verification padding can exceed the real-token limit. Never
                # truncate. All rows stay in one owner allocation, including
                # finished rows which have not been materialized yet.
                capacity=max(end,pool.shape[1]*2)
                grown=pool.new_empty((self.response_count,capacity,*pool.shape[2:]))
                grown[:,:start].copy_(pool[:,:start])
                self.buffers[name]=pool=grown;self.capacities[name]=capacity
            pool[:,start:end].index_copy_(0,owners,chunk)
            self.lengths[name]=end

    def mark_finished(self, row):
        if row in self.finished:raise ValueError('history row finished twice')
        self.finished[row]=self.lengths.copy()

    def finalize(self):
        """Pack used histories once; output views own only this compact storage."""
        for row in range(self.response_count):
            if row not in self.finished:self.mark_finished(row)
        result=[{} for _ in range(self.response_count)]
        for name,pool in self.buffers.items():
            lengths=[self.finished[row][name] for row in range(self.response_count)]
            total=sum(lengths)
            # One GPU gather per field, no per-response copies or GPU scalar reads.
            starts=torch.tensor([row*pool.shape[1]-sum(lengths[:row]) for row in range(self.response_count)],
                                device=pool.device,dtype=torch.long)
            counts=torch.tensor(lengths,device=pool.device,dtype=torch.long)
            indices=torch.repeat_interleave(starts,counts,output_size=total)+torch.arange(total,device=pool.device)
            packed=pool.flatten(0,1).index_select(0,indices)
            offset=0
            for row,length in enumerate(lengths):
                result[row][name]=packed[offset:offset+length]
                offset+=length
        self.buffers.clear()
        return result
