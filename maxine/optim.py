# Responsible for updating the parameters of the model during training based on computed gradients

class SDG(object):
    def __init__(self, params, lr): self.params, self.lr = list(params), lr
    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr
    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
