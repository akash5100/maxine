import unittest
import torch

from nn.loader.dataloader import DataLoaders


class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        # test with more like a 3x3 image
        self.x_train = torch.tensor(
            [
                [[1, 2, 1], [1, 2, 1], [1, 2, 1]],
                [[3, 4, 1], [1, 2, 1], [1, 2, 1]],
                [[5, 6, 1], [1, 2, 1], [1, 2, 1]],
                [[7, 8, 1], [1, 2, 1], [1, 2, 1]],
            ]
        )
        self.y_train = torch.tensor([0, 1, 0, 1])
        self.batch_size = 2
        self.dls = DataLoaders(self.x_train, self.y_train, self.batch_size)

    def test_data_loader_length(self):
        self.assertEqual(len(self.dls), len(self.x_train) // self.batch_size)

    def test_data_loader_iteration(self):
        batches = list(self.dls)
        self.assertEqual(len(batches), len(self.x_train) // self.batch_size)
        for x_batch, y_batch in batches:
            self.assertEqual(len(x_batch), self.batch_size)
            self.assertEqual(len(y_batch), self.batch_size)

    def test_validation_data(self):
        x_val = torch.tensor([[9, 10], [11, 12]])
        y_val = torch.tensor([0, 1])

        dls = DataLoaders(
            self.x_train, self.y_train, self.batch_size, x_val=x_val, y_val=y_val
        )
        vdls = dls.get_validation_data()
        self.assertIsNotNone(vdls)


if __name__ == "__main__":
    unittest.main()
