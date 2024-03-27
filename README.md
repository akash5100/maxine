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


Here is a [demo training mnist dataset](https://github.com/akash5100/maxine/blob/main/examples/mnist_maxine.ipynb)

## NN Module

```py
import maxine.nn as nn

x = torch.randn(60000, 1, 28, 28)
x = x.view(x.size(0), -1)

ih = nn.Linear(784, 10)
ih.forward(x)
```


## Losses

```py
TODO
```


## Metrics

```py
from maxine.metrics import accuracy

a = torch.randn(3,5).normal_(0,1)
b = torch.Tensor([2, 1, 5])
accuracy(a, b)
# output:
# tensor(0.3333)
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
