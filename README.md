I project to teach me the working/implementation of Neural net models under-the-hood

Well with just one help:
- I used `torch.Tensor`
- to calculate Gradient I used `requires_grad_` (from PyTorch)

## Goal 1 -- Implement `Learner` from scratch
Here is my idea, top of the list represents class/function and below it represents its parameters. 
- [x] Learner(DataLoaders, model)
    - fit(num_epochs)
    - train_epoch
      1. calculate gradients
      2. step:: update weights
      3. set gradients to zero (so they don't add up)
    - batch_accuracy:: Calculate accuracy to train the model
    - validate_epoch:: Accuracy calculated with the validation dataset

- [x] DataLoaders(x_train, y_train, batch_size, shuffle=True, x_val=None, y_val=None)
    - flatten_images( images )
    - `__iter__`:: returns `x_train` and `y_train` in batches format.
    - get_validation_data:: returns `x_val` and `y_val` in batches format.

- [x] MNISTModel(size, learning rate)
  - linear1(xb):: Simple matrix multiplication `xb@weights+bias`
  - mnist_loss(preds, target)
  - calc_grad(xb, yb, model):
    - make preds,
    - compare with labels,
    - backdrop <-- PyTorch help
  - step:: update parameters <-- PyTorch help
  - zero_grad:: set grads to zero <-- PyTorch help


## Goal 2 -- Train NN in MNIST dataset
See [demo](https://github.com/akash5100/nn/blob/main/nn/demo.ipynb)

## Installation
Create a python environment, then:
```bash
pip install -r requirements.txt
```
Install the project:
```bash
pip install -e .
```
Run unittest:
```bash
python -m unittest discover
```
