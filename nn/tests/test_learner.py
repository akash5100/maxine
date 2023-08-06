import unittest
import torch

from nn.learner.learner import Learner
from nn.model.linear import MNISTModel
from nn.loader.dataloader import DataLoaders


class TestLearner(unittest.TestCase):
    def setUp(self):
        # Create mock data
        self.x_train = torch.rand(100, 28 * 28)
        self.y_train = torch.randint(0, 10, (100,))
        self.x_val = torch.rand(20, 28 * 28)
        self.y_val = torch.randint(0, 10, (20,))

        # Create dataloader, model, optimizer etc.
        self.dataloader = DataLoaders(
            self.x_train,
            self.y_train,
            batch_size=32,
            x_val=self.x_val,
            y_val=self.y_val,
        )

        self.model = MNISTModel(28 * 28, lr=0.1)
        self.loss_fn = MNISTModel.mnist_loss

    def test_learner_fit(self):
        learner = Learner(self.dataloader, self.model, self.loss_fn)

        learner.fit(num_epochs=3)

        # Assert model trained
        self.assertNotAlmostEqual(self.model.weights.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
