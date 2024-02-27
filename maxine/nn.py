# This module is used as building blocks for all neural network modules
from maxine.torch_imports import Tensor, tensor, torch, np
from typing import Union, Tuple, List
from functools import reduce
# https://docs.python.org/3.0/library/functools.html?highlight=reduce#functools.reduce


class Parameter:
    "Initializes trainable weights"
    def __init__(self, *sz: Union[int, Tuple, List]) -> None:
        t_sz = reduce(lambda x,y: x*y, sz) if sz else 1
        self.w = tensor(np.random.uniform(-1., 1., size=sz)/np.sqrt(t_sz), requires_grad=True, dtype=torch.float32)

    @property
    def shape(self): return self.w.shape
    @property
    def weight(self) -> Tensor: return self.w
    @property
    def grad(self): return self.w.grad
    def __repr__(self) -> str: return f"{self.w}"


class Module(object):
    def forward(self, x): raise NotImplementedError
    def backward(self): raise NotImplementedError
    def parameters(self): raise NotImplementedError # need for optim
    def zero_grad(self): raise NotImplementedError  # need for optim


class Linear(Module):
    def __init__(self, in_: int, out_: int) -> None: self.w, self.b = Parameter(in_, out_), Parameter(out_)
    def __repr__(self) -> str: return f'{self.w}'
    def forward(self, x: Tensor) -> Tensor: return x@self.w.weight + self.b.weight
    def parameters(self) -> Tensor: return self.w.weight, self.b.weight
    def zero_grad(self): self.w.weight.grad, self.b.weight.grad = None, None

class Dropout(Module):
    def __init__(self, p) -> None: self.p = p
    def forward(self, x: Tensor):
        mask = (x.div_(1-self.p)) * (x.new(*x.shape).bernoulli_(self.p))
        return mask * x

class Embedding(Module):
    pass