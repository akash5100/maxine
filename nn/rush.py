import torch
from fastai.vision.all import *

path = untar_data(URLs.MNIST_SAMPLE)

threes = (path / "train" / "3").ls().sorted()
sevens = (path / "train" / "7").ls().sorted()

# Training Image to tensors
three_tensors = torch.stack(
    [tensor(Image.open(o)) for o in (path / "train" / "3").ls()]
)
seven_tensors = torch.stack(
    [tensor(Image.open(o)) for o in (path / "train" / "7").ls()]
)
stacked_threes = three_tensors.float() / 255
stacked_sevens = seven_tensors.float() / 255

# Validation Image to tensors
valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path / "valid" / "3").ls()])
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path / "valid" / "7").ls()])
valid_3_tens = valid_3_tens.float() / 255
valid_7_tens = valid_7_tens.float() / 255
""


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


""


def init_params(size, std=1.0):
    return (torch.randn(size) * std).requires_grad_()


""
# Mnist has image of size 28*28
weights = init_params(28 * 28)
bias = init_params(1)
""


# something to make preds
def linear1(xb):
    return xb @ weights + bias


train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28 * 28)
train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28 * 28)
valid_y = tensor([1] * len(valid_3_tens) + [0] * len(valid_7_tens)).unsqueeze(1)


dset = list(zip(train_x, train_y))
valid_dset = list(zip(valid_x, valid_y))
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

# Make preds
preds = linear1(train_x)
""


def mnist_loss(preds, target):
    """
    preds = predictions, target = labels
    so when,
    labels == 1, which means correct prediction
        -> then we return the loss 1-preds
    else: the prediction is wrong, obviously
        -> then we return preds
    """
    preds = sigmoid(preds)
    return torch.where(target == 1, 1 - preds, preds).mean()


# Template of Training
"""
for x,y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()
    parameters -= parameters.grad * lr
    
batch =train_x[:4]
preds = linear1(batch)
loss = mnist_loss(preds, train_y[:4])
"""


def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()


class BasicOptim:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None


weights = torch.randn(784, 1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

opt = BasicOptim([weights, bias], lr=0.1)
