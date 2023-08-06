import unittest
import torch

from nn.loader.dataloader import DataLoaders
from nn.model.linear import MNISTModel

class TestMNISTModel(unittest.TestCase):
    def setUp(self):
        """
        To the MNISTModel, we need to pass:
        - DataLoaders :c
        - model
        - loss function
        - learning rate
        """
        pass

    def test_model(self):
        pass

if __name__ == "__main__":
    unittest.main()
