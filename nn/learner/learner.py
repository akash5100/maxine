from nn.loader.dataloader import DataLoaders

class Learner:
    def __init__(self, dls: DataLoaders, model:callable, optimizer, loss_func, metrics):
        self.model = model
        self.dls = dls
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics = metrics

    def train_epoch(self, model, dl):
        # for xb, yb in dl:
        #     self.calc_grad(xb, yb, model)
        #     self.step()
        #     self.zero_grad()
        pass

    def fit(self, epoch, lr):
        """
        assume, we already have the below info:
            - dls
            - model
            - loss func
            - validate_func
        for e in range(epoch):
            self.train_epoch(model)
            print(validate_epoch(model), end=" ")
        """
        pass

    def validate_epoch(self):
        pass