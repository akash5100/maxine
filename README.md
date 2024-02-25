# Maxine

Named after Matrix Chain (Matrix Multiplication in series), after all thats what Neural Nets are.

This project is a fun implementation of Neural Network to understand the under-the-hood working. 

- Optim class -- Optimizer
- Model class -- Neural Network model
- MLOPS -- Operators required for Deep learning, like Sigmoid, ReLU etc

Here are the steps to follow:

1. Write a Model
2. define Loss function
3. Create an Optimizer such as SGD or Adam
4. Load the data to tensor in a Data Loader
5. write a Training Loop
6. train it
7. Train MNIST subset- for eg, 3 vs 7
8. Train full MNIST
9. Utilities-- Saving Loading pickel file
10. Expand

Libraries used-
 - Numpy
 - Pytorch's Tensor, only for calculating Gradient
 - tqdm

A much later goal in this project-
 - remove pytorch dependency (gradient)
 - Create tensor module


Here is a [demo training mnist dataset](https://github.com/akash5100/nn/blob/main/nn/demo.ipynb)


## NN Module

```py
>>> import maxine.nn as nn
>>> ih = nn.Linear(784, 10)

>>> ih.w.shape
torch.Size([784, 10])
>>> ih.b.shape
torch.Size([10])

>>> x_train = torch.randn(60000, 1, 28, 28)
>>> x_train = x_train.view(x_train.size(0), -1)
>>> x_train.shape
torch.Size([60000, 784])

>>> ih.forward(x_train)
tensor([[ 0.3503,  0.0352,  0.5296,  ..., -0.3257, -0.0389, -0.2227],
        [ 0.1385,  0.4990,  0.3538,  ...,  0.0127, -0.0094, -0.0592],
        [-0.0821,  0.0817,  0.3141,  ..., -0.1158,  0.1390, -0.5184],
        ...,
        [-0.0211,  0.0705, -0.0290,  ..., -0.2370, -0.1585, -0.1410],
        [ 0.1859,  0.2334,  0.4287,  ..., -0.1808, -0.3707, -0.3433],
        [ 0.3480,  0.5040,  0.2392,  ..., -0.0928,  0.2971, -0.0568]],
       grad_fn=<AddBackward0>)
>>>
```


## Losses

```py
TODO
```


## Metrics

```py
>>> from maxine.metrics import accuracy
>>> from maxine.accu
>>> a = torch.randn(3,5).normal_(0,1)
>>> a
tensor([[ 1.7868e-01,  9.1171e-01,  9.0093e-01,  3.0340e-01, -5.1710e-01],
        [ 4.1547e-01,  1.3547e+00, -5.9686e-01,  9.2384e-01,  2.4055e-02],
        [-1.5665e+00,  1.0830e+00,  9.0813e-04, -6.7731e-01,  1.3082e-01]])
>>> b = torch.Tensor([2, 1, 5])
>>> accuracy(a, b)
tensor(0.3333)
```


## Installation:


```bash
pip install -r requirements.txt
```
```bash
pip install -r requirements-dev.txt
```
Install the project:
```bash
pip install -e .
```
