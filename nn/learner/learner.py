import torch
from nn.loader.dataloader import DataLoaders
from nn.utils import sigmoid


class Learner:
    def __init__(self, dls: DataLoaders, model: callable, loss_func=None, metrics=None):
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.metrics = metrics

    def fit(self, num_epochs):
        """
        assume, we already have the below info:
            - dls
            - model
            - loss func
            - validate_func
        """
        for e in range(num_epochs):
            self.train_epoch()
            val_acc = self.validate_epoch(self.model.linear1)
            print(f"Epoch {e+1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}")

    def train_epoch(self):
        for xb, yb in self.dls:
            self.model.calc_grad(xb, yb, model=self.model.linear1)
            self.model.step()
            self.model.zero_grad()

    def batch_accuracy(self, xb, yb):
        """calculate preds
        use sigmoid to ... (you know why we use sigmoid)
        if >0.5 return its mean"""
        preds = sigmoid(xb)
        correct = (preds > 0.5) == yb
        return correct.float().mean()

    def validate_epoch(self, model):
        accs = []
        for xb, yb in self.dls.get_validation_data():
            acc = self.batch_accuracy(model(xb), yb)
            accs.append(acc)
        return round(torch.stack(accs).mean().item(), 4)

    def predict(self):
        # todo
        pass

    def save(self):
        # todo
        pass

    def load(self):
        # todo
        pass
