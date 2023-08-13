import torch
from torch import exp, randn


def sigmoid(x):
    return 1 / (1 + exp(-x))


def init_params(size, std):
    return (randn(size) * std).requires_grad_()


def relu(x):
    """Relu relu relu relu"""
    return torch.max(x, torch.tensor(0.0))
