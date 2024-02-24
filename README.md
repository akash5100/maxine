# Maxine

Named after Matrix Chain (Matrix Multiplication in series), after all thats what Neural Nets are.

This project is a fun implementation of Neural Network to understand the under-the-hood working. 

- Optim class -- Optimizer
- Model class -- Neural Network model
- MLOPS -- Operators required for Deep learning, like Sigmoid, ReLU etc

Here are the steps to follow:

1.  Write a Model
2. Create an Optimizer such as SGD or Adam
3. define Loss function
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
 - remove pytorch dependency


Here is a [demo training mnist dataset](https://github.com/akash5100/nn/blob/main/nn/demo.ipynb)

Installation guide:

```bash
pip install -r requirements.txt
```
Install the project:
```bash
pip install -e .
```
