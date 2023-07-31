import numpy as np

__all__ = ['DataLoaders']

class DataLoaders:
    def __init__(self, x_train, y_train, batch_size, shuffle=True, x_val=None, y_val=None):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x_val = x_val
        self.y_val = y_val
        
    def __len__(self):
        """Returns total number of data in single batch."""
        return len(self.x_train) // self.batch_size
    
    def __iter__(self):
        indices = np.arange(len(self.x_train))
        if self.shuffle:
            # perform shuffling
            np.random.shuffle(indices)
        
        # Implement minibatch iteration and return a list of mini-batches
        batches = []
        for i in range(0, len(self.x_train), self.batch_size):
            x_batch = self.x_train[indices[i : i+self.batch_size]]
            y_batch = self.y_train[indices[i : i+self.batch_size]]
            batches.append((x_batch, y_batch))

        return iter(batches)

    def __str__(self):
        train_data = list(zip(self.x_train, self.y_train))
        return f"{train_data}"

    def get_validation_data(self):
        if self.x_val is not None and self.y_val is not None:
            return list(zip(self.x_val, self.y_val))
        else:
            return None