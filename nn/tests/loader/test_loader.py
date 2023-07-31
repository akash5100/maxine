import unittest
import numpy as np

from nn.loader.loader import DataLoaders

class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        self.x_train = np.arange(100)
        self.y_train = np.arange(100) * 2
        self.batch_size = 10

    def test_data_loader_length(self):
        dls = DataLoaders(self.x_train, self.y_train, self.batch_size)
        self.assertEqual(len(dls), len(self.x_train) // self.batch_size)

    def test_data_loader_iteration(self):
        dls = DataLoaders(
            self.x_train, self.y_train, self.batch_size, shuffle=False
        )

        batches = list(iter(dls))
        self.assertEqual(len(batches), len(self.x_train) // self.batch_size)
        for x_batch, y_batch in batches:
            self.assertEqual(len(x_batch), self.batch_size)
            self.assertEqual(len(y_batch), self.batch_size)

        # Test shuffled iteration
        shuffled_dls= DataLoaders(self.x_train, self.y_train, self.batch_size, shuffle=True)
        shuffled_batches = list(iter(shuffled_dls))

        # Check that at least one batch is different
        is_different_batch_found = any((x1 != x2).any() or (y1 != y2).any() for (x1, y1), (x2, y2) in zip(batches, shuffled_batches))
        self.assertTrue(is_different_batch_found)

    def test_validation_data(self):
        x_val = np.arange(100, 120)
        y_val = np.arange(200, 220)

        dls = DataLoaders(
            self.x_train, self.y_train, self.batch_size, x_val=x_val, y_val=y_val
        )
        val_data = dls.get_validation_data()

        self.assertIsNotNone(val_data)
        self.assertEqual(len(val_data), len(x_val))
        self.assertEqual(len(val_data[0]), 2)  # Each element should be a tuple (x_val, y_val)

        # Check if the validation data corresponds to the input x_val and y_val
        for i, (x_val_item, y_val_item) in enumerate(val_data):
            self.assertEqual(x_val_item, x_val[i])
            self.assertEqual(y_val_item, y_val[i])

if __name__ == '__main__':
    unittest.main()
