from torch import exp, randn 

def sigmoid(x):
    return 1/(1+exp(-x))

def init_params(size, std):
    return (randn(size)*std).requires_grad_()
