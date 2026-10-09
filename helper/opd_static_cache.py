"""Growable flat HF KV cache: suffix append, geometric growth, row remap.

Attention sees a valid prefix. Only geometric growth replaces its pool.
Finished rows use disjoint tail-to-hole copies; unchanged survivors are untouched.
Generic repeat/select still uses scratch for arbitrary aliasing permutations.
Pools persist across rollouts; logical history resets without exposing stale KV.
"""
import torch
import weakref
from transformers.cache_utils import Cache
try:
    from transformers.cache_utils import CacheLayerMixin
except ImportError:
    CacheLayerMixin = object


class StaticAppendLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, capacity, owner=None):
        self.key_pool = self.value_pool = None
        self.length = self.batch_size = self.batch_capacity = 0
        self._owner_ref = weakref.ref(owner) if owner is not None else None
        self.stats = dict(pool_allocations=0, full_kv_reallocations=0,
                          full_history_copies=0, full_history_copy_bytes=0,
                          row_compactions=0, row_compaction_bytes=0)
        super().__init__()
        self.capacity = capacity
        self.is_initialized = False

    @property
    def owner(self):
        # The cache owns layers; a strong backlink would delay freeing ALL GPU
        # pools until cyclic GC after the rollout returns.
        return self._owner_ref() if self._owner_ref is not None else None

    @property
    def keys(self):
        return None if self.key_pool is None else self.key_pool[:self.batch_size, :, :self.length, :]

    @keys.setter
    def keys(self, value):
        if value is not None:
            self._set_view(value, True)

    @property
    def values(self):
        return None if self.value_pool is None else self.value_pool[:self.batch_size, :, :self.length, :]

    @values.setter
    def values(self, value):
        if value is not None:
            self._set_view(value, False)

    def _set_view(self, value, key):
        pool = self.key_pool if key else self.value_pool
        if pool is None:
            raise RuntimeError('cache assignment before initialization')
        if value.untyped_storage().data_ptr() != pool.untyped_storage().data_ptr():
            raise RuntimeError('use batch_select_indices for row remap; external KV replacement is unsupported')
        self.length, self.batch_size = value.shape[-2], value.shape[0]

    def _capacity_for(self, needed):
        chunk = self.owner.chunk_size if self.owner is not None else 256
        return max(self.capacity, ((needed + chunk - 1) // chunk) * chunk)

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size = key_states.shape[0]
        self.batch_capacity = max(self.batch_size, self.owner.batch_capacity if self.owner is not None else 0)
        self.capacity = self._capacity_for(key_states.shape[-2])
        shape = (self.batch_capacity, key_states.shape[1], self.capacity, key_states.shape[-1])
        self.key_pool = key_states.new_empty(shape)
        self.value_pool = value_states.new_empty(shape)
        self.stats['pool_allocations'] += 1
        self.is_initialized = True

    def _grow(self, needed, batch_needed=None):
        batch_needed = self.batch_size if batch_needed is None else batch_needed
        if needed <= self.capacity and batch_needed <= self.batch_capacity:
            return
        capacity = self._capacity_for(max(needed, 2*self.capacity)) if needed > self.capacity else self.capacity
        rows = max(batch_needed, self.batch_capacity)
        shape = (rows, self.key_pool.shape[1], capacity, self.key_pool.shape[-1])
        old_keys, old_values = self.keys, self.values
        keys, values = self.key_pool.new_empty(shape), self.value_pool.new_empty(shape)
        if self.length:
            keys[:self.batch_size, :, :self.length].copy_(old_keys)
            values[:self.batch_size, :, :self.length].copy_(old_values)
        self.stats['pool_allocations'] += 1
        self.stats['full_kv_reallocations'] += 1
        if self.length:
            self.stats['full_history_copies'] += 1
            self.stats['full_history_copy_bytes'] += old_keys.numel()*old_keys.element_size()+old_values.numel()*old_values.element_size()
        self.key_pool, self.value_pool = keys, values
        self.capacity, self.batch_capacity = capacity, rows

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if key_states.shape[0] != self.batch_size:
            raise ValueError('KV batch differs from live request count')
        end = self.length + key_states.shape[-2]
        self._grow(end)
        self.key_pool[:self.batch_size, :, self.length:end].copy_(key_states)
        self.value_pool[:self.batch_size, :, self.length:end].copy_(value_states)
        self.length = end
        return self.keys, self.values

    def get_seq_length(self): return self.length
    def get_max_cache_shape(self): return self.capacity
    def get_max_length(self): return self.capacity
    def get_mask_sizes(self, query_length): return self.length+query_length, 0
    def crop(self, length): self.length = max(0, min(self.length, length if length >= 0 else self.length+length))
    def reset(self): self.length = 0

    def batch_select_indices(self, indices):
        new_batch = indices.numel()
        self._grow(self.length, new_batch)
        chunk = self.owner.chunk_size if self.owner is not None else 256
        # Gather before overwriting: CUDA uses two stream-ordered kernels and
        # shared live-prefix scratch; CPU uses bounded token tiles. Both preserve
        # arbitrary permutations and duplicate indices without aliasing races.
        if self.key_pool.is_cuda:
            from helper.opd_kv_kernels import remap_rows
            count=2*new_batch*self.key_pool.shape[1]*self.length*self.key_pool.shape[-1]
            key=('cuda_live',self.key_pool.dtype,self.key_pool.device)
            scratch=self.owner._scratch.get(key) if self.owner is not None else None
            if scratch is None or scratch.numel()<count:
                scratch=self.key_pool.new_empty(1<<max(0,(max(1,count)-1).bit_length()))
                if self.owner is not None:self.owner._scratch[key]=scratch
            remap_rows(self.key_pool,self.value_pool,indices,self.length,scratch)
        else:
            for pool in (self.key_pool, self.value_pool):
                shape = (new_batch, pool.shape[1], min(chunk, max(1,self.length)), pool.shape[-1])
                scratch = self.owner.scratch(pool, shape) if self.owner is not None else pool.new_empty(shape)
                for start in range(0,self.length,chunk):
                    width = min(chunk,self.length-start)
                    tile = scratch[:, :, :width]
                    torch.index_select(pool[:self.batch_size, :, start:start+width],0,indices,out=tile)
                    pool[:new_batch, :, start:start+width].copy_(tile)
        self.stats['row_compactions'] += 1
        moved=new_batch*self.key_pool.shape[1]*self.length*self.key_pool.shape[-1]*(self.key_pool.element_size()+self.value_pool.element_size())
        self.stats['row_compaction_bytes'] += moved
        if moved:
            self.stats['full_history_copies'] += 1
            self.stats['full_history_copy_bytes'] += moved
        self.batch_size = new_batch

    def batch_repeat_interleave(self, repeats):
        if repeats < 1: raise ValueError('positive repeats required')
        if repeats != 1:
            indices = torch.arange(self.batch_size,device=self.key_pool.device).repeat_interleave(repeats)
            self.batch_select_indices(indices)

    def swap_remove(self,new_batch,sources,destinations):
        # Sources are disjoint live tail rows; destinations are holes below
        # new_batch. No gather scratch/barrier or survivor-wide copy required.
        count=sources.numel()
        if count and self.length:
            if self.key_pool.is_cuda:
                from helper.opd_kv_kernels import move_tail_rows
                move_tail_rows(self.key_pool,self.value_pool,sources,destinations,self.length)
            else:
                for source,destination in zip(sources,destinations):
                    self.key_pool[destination,:,:self.length].copy_(self.key_pool[source,:,:self.length])
                    self.value_pool[destination,:,:self.length].copy_(self.value_pool[source,:,:self.length])
        moved=count*self.key_pool.shape[1]*self.length*self.key_pool.shape[-1]*(self.key_pool.element_size()+self.value_pool.element_size())
        self.stats['row_compactions']+=1
        self.stats['row_compaction_bytes']+=moved
        self.stats['full_history_copy_bytes']+=moved
        if moved:self.stats['full_history_copies']+=1
        self.batch_size=new_batch


class OPDStaticCache(Cache):
    def __init__(self, capacity=256, *, batch_capacity=0, chunk_size=256):
        if capacity < 1 or chunk_size < 1 or batch_capacity < 0:
            raise ValueError('positive KV capacity/chunk and nonnegative batch capacity required')
        try: super().__init__(layers=[])
        except TypeError:
            super().__init__()
            self.layers = []
        self.capacity, self.batch_capacity, self.chunk_size = capacity,batch_capacity,chunk_size
        self._scratch = {}
        self.kv_rows_moved=0

    def begin_rollout(self,batch,batch_capacity):
        self.batch_capacity=max(self.batch_capacity,batch_capacity)
        self.kv_rows_moved=0
        for layer in self.layers:
            layer.length=0;layer.batch_size=batch
            for key in layer.stats:layer.stats[key]=0

    def end_rollout(self,max_retained_tokens=0):
        for layer in self.layers:layer.length=layer.batch_size=0
        if max_retained_tokens and any(layer.capacity>max_retained_tokens for layer in self.layers):
            self.layers.clear();self._scratch.clear()

    def scratch(self,pool,shape):
        rows=shape[0]
        key = (pool.device,pool.dtype,shape[1],shape[-1])
        scratch = self._scratch.get(key)
        if scratch is None or any(a < b for a,b in zip(scratch.shape,shape)):
            shape = (max(shape[0],self.batch_capacity),shape[1],self.chunk_size,shape[-1])
            scratch = pool.new_empty(shape)
            self._scratch[key] = scratch
        return scratch[:rows]

    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        while len(self.layers) <= layer_idx:
            self.layers.append(StaticAppendLayer(self.capacity,self))
        return self.layers[layer_idx].update(key_states,value_states)

    def get_seq_length(self,layer_idx=0): return self.layers[layer_idx].length if layer_idx < len(self.layers) else 0
    def get_mask_sizes(self,query_length,layer_idx=0): return self.get_seq_length(layer_idx)+query_length,0
    def get_max_cache_shape(self,layer_idx=0): return self.layers[layer_idx].capacity if layer_idx < len(self.layers) else self.capacity
    def crop(self,length):
        for layer in self.layers: layer.crop(length)
    def batch_repeat_interleave(self,repeats):
        for layer in self.layers: layer.batch_repeat_interleave(repeats)
    def batch_select_indices(self,indices):
        for layer in self.layers: layer.batch_select_indices(indices)
        # Generic selection copies every output row; prefill repeat calls layer
        # methods directly and is intentionally excluded from finish-row counts.
        self.kv_rows_moved+=indices.numel()
    def swap_remove(self,new_batch,sources,destinations):
        for layer in self.layers:layer.swap_remove(new_batch,sources,destinations)
        self.kv_rows_moved+=sources.numel()
    def __getitem__(self,index): return self.layers[index].keys,self.layers[index].values
    def __iter__(self):
        for layer in self.layers: yield layer.keys,layer.values
    def __len__(self): return len(self.layers)
    def __bool__(self): return bool(self.layers) and self.get_seq_length()>0

    def statistics(self):
        names = ('pool_allocations','full_kv_reallocations','full_history_copies','full_history_copy_bytes','row_compactions','row_compaction_bytes')
        totals = {key:sum(layer.stats[key] for layer in self.layers) for key in names}
        totals['kv_cache_bytes'] = sum(pool.numel()*pool.element_size() for layer in self.layers for pool in (layer.key_pool,layer.value_pool) if pool is not None)
        totals['kv_compaction_workspace_bytes'] = sum(x.numel()*x.element_size() for x in self._scratch.values())
        totals['kv_rows_moved']=self.kv_rows_moved
        totals['kv_history_copy_bytes']=totals['full_history_copy_bytes']
        return totals


def swap_remove_plan(finished):
    """Host metadata already provided by the sole scheduling packet."""
    new_batch=len(finished)-sum(bool(x) for x in finished)
    holes=[i for i in range(new_batch) if finished[i]]
    tail=[i for i in range(len(finished)-1,new_batch-1,-1) if not finished[i]]
    keep=list(range(new_batch))
    for destination,source in zip(holes,tail):keep[destination]=source
    return keep,tail,holes


def persistent_cache(model,name,batch,responses,device,dtype):
    cache=getattr(model,name,None)
    signature=(str(device),dtype)
    if cache is None or getattr(cache,'runtime_signature',None)!=signature:
        cache=OPDStaticCache(256,batch_capacity=batch*responses)
        cache.runtime_signature=signature;setattr(model,name,cache)
    cache.begin_rollout(batch,batch*responses)
    return cache
