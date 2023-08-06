import torch

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def init_params(size, std):
    return (torch.randn(size)*std).requires_grad_()

class BasicOptim:
    def __init__(self,params,lr): 
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params: 
            p.data -= p.grad.data * self.lr

    def zero_grad(self):
        for p in self.params: 
            p.grad = None


class MNISTModel:
    def __init__(self, size, lr, std=1.0):
        self.weights = init_params(size, std)
        self.bias = init_params(1, std)
        self.optimizer = BasicOptim([self.weights, self.bias], lr)

    def linear1(self, xb):
        """xb: x_train batch."""
        return xb@self.weights + self.bias
    
    def mnist_loss(self, preds, target):
        """
        preds = predictions, target = labels
        so when, 
        labels == 1, which means correct prediction 
            -> then we return the loss, i.e, 
               (1 - `correct` preds)
        else: the prediction is wrong, obviously
            -> then we return preds
        """
        preds = sigmoid(preds)
        return torch.where(target == 1, 1-preds, preds).mean()
    
    def calc_grad(self, xb, yb, model):
        """
        make preds,
        compare with labels,
        backprop :)
        """
        preds = model(xb)
        loss = self.mnist_loss(preds=preds, target=yb)
        loss.backward()
        
    def batch_accuracy(xb, yb):
        """calculate preds
        use sigmoid to ... (you know why we use sigmoid)
        if >0.5 return its mean"""
        preds = sigmoid(xb)
        correct = (preds > 0.5) == yb
        return correct.float().mean()
    
    def step(self):
        # Use the optimizer to update parameters
        self.optimizer.step()

    def zero_grad(self):
        # use the optimizer to zero grads
        self.optimizer.zero_grad()
