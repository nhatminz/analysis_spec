"""OPD causal mask: fixed tiles, runtime batch/query/past and no host scalars."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['PAST','ROWS','COLS','PS0','PS1'])
def _causal(MASK,PAD,PAST,ROWS,COLS,PS0,PS1,MIN:tl.constexpr,HAS_PAD:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    visible=col<=PAST+row%ROWS
    if HAS_PAD:
        padding=tl.load(PAD+row//ROWS*PS0+col*PS1,col<COLS,other=0)
        visible=visible&~padding
    tl.store(MASK+row*COLS+col,tl.where(visible,0.,MIN),col<COLS)


def causal_mask(out,past,padding=None):
    b,_,q,columns=out.shape
    _causal[(b*q,triton.cdiv(columns,256))](out,padding,past,q,columns,
        *(padding.stride() if padding is not None else (0,0)),
        torch.finfo(out.dtype).min,padding is not None,256,num_warps=4)
