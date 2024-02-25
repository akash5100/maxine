# This module is used as building blocks for all neural network modules
from maxine.torch_imports import Tensor, tensor, torch, np
from typing import Union, Tuple, List
from functools import reduce
# https://docs.python.org/3.0/library/functools.html?highlight=reduce#functools.reduce



class Module(object):
    def forward(self, *x): raise NotImplementedError
    def backward(self): raise NotImplementedError


class Parameters:
    "Initializes trainable weights"
    def __init__(self, *sz: Union[int, Tuple, List]) -> None:
        t_sz = reduce(lambda x,y: x*y, sz) if sz else 1
        self.w = tensor(np.random.uniform(-1., 1., size=sz)/np.sqrt(t_sz), requires_grad=True, dtype=torch.float32)

    @property
    def shape(self): return self.w.shape
    @property
    def T(self) -> Tensor: return self.w
    def __repr__(self) -> str: return f"{self.w}"


class Linear(Module):
    def __init__(self, in_: int, out_: int) -> None:
        self.w = Parameters(in_, out_)
        self.b = Parameters(out_)

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data: y = x@w+b"""
        w, b = self.w.T, self.b.T
        return torch.matmul(x,w) + b

    def __repr__(self) -> str: return f'{self.w}'
