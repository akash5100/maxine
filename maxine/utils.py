from maxine.torch_imports import Tensor

def flatten_check(inp, targ) -> Tensor:
    "Check that `inp` and `targ` have the same number of elements and flatten them."
    inp,targ = inp.contiguous().view(-1),targ.contiguous().view(-1)
    if len(inp) != len(targ): raise ValueError(f"inp len: {len(inp)}; targs len: {len(targ)}")
    return inp,targ
