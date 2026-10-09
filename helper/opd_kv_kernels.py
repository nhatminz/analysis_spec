"""KV row remap via reusable LIVE-prefix scratch and ordered gather/scatter.

Two stream-ordered kernels are the global read-before-write barrier. The scratch
is shared across all layers and grows geometrically; never copied unused KV
capacity, never replaces a KV pool on finish, no host synchronization.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N','H','LENGTH','CAPACITY','D'])
def _gather(K,V,INDICES,SCRATCH,N,H,LENGTH,CAPACITY,D,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=x//(H*LENGTH*D)
    col=x%(H*LENGTH*D)
    source=tl.load(INDICES+row,x<N,other=0).to(tl.int64)
    offset=(col//(LENGTH*D)*CAPACITY+col//D%LENGTH)*D+col%D
    ptr=source*(H*CAPACITY*D)+offset
    k=tl.load(K+ptr,x<N,other=0);v=tl.load(V+ptr,x<N,other=0)
    tl.store(SCRATCH+x,k,x<N);tl.store(SCRATCH+N+x,v,x<N)


@triton.jit(do_not_specialize=['N','H','LENGTH','CAPACITY','D'])
def _scatter(SCRATCH,K,V,N,H,LENGTH,CAPACITY,D,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=(x//(H*LENGTH*D)).to(tl.int64)
    col=x%(H*LENGTH*D)
    offset=(col//(LENGTH*D)*CAPACITY+col//D%LENGTH)*D+col%D
    ptr=row*(H*CAPACITY*D)+offset
    k=tl.load(SCRATCH+x,x<N,other=0);v=tl.load(SCRATCH+N+x,x<N,other=0)
    tl.store(K+ptr,k,x<N);tl.store(V+ptr,v,x<N)


def remap_rows(keys,values,indices,length,scratch):
    _,heads,capacity,dim=keys.shape
    n=indices.numel()*heads*length*dim
    if n:
        grid=(triton.cdiv(n,1024),)
        _gather[grid](keys,values,indices,scratch,n,heads,length,capacity,dim,1024,num_warps=4)
        _scatter[grid](scratch,keys,values,n,heads,length,capacity,dim,1024,num_warps=4)


@triton.jit(do_not_specialize=['N','H','LENGTH','CAPACITY','D'])
def _move_tail(K,V,SRC,DST,N,H,LENGTH,CAPACITY,D,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=x//(H*LENGTH*D);col=x%(H*LENGTH*D)
    source=tl.load(SRC+row,x<N,other=0).to(tl.int64)
    destination=tl.load(DST+row,x<N,other=0).to(tl.int64)
    offset=(col//(LENGTH*D)*CAPACITY+col//D%LENGTH)*D+col%D
    k=tl.load(K+source*H*CAPACITY*D+offset,x<N,other=0)
    v=tl.load(V+source*H*CAPACITY*D+offset,x<N,other=0)
    tl.store(K+destination*H*CAPACITY*D+offset,k,x<N)
    tl.store(V+destination*H*CAPACITY*D+offset,v,x<N)


def move_tail_rows(keys,values,sources,destinations,length):
    _,h,capacity,d=keys.shape;n=sources.numel()*h*length*d
    if n:_move_tail[(triton.cdiv(n,1024),)](keys,values,sources,destinations,n,h,length,capacity,d,1024,num_warps=4)
