# NN
Creating Neural Network to train MNIST data from Scratch.

Well with just one help:
- Calculate Gradient (from pytorch)

## Goal 1 -- Implement `Learner` from scratch

Here is my idea, top of the list represents class/function and below it represents its parameters. 

- Learner
    - [x] DataLoaders
        - [x] collection (iter)
            - [x] x_train (actual data)
            - [x] y_train (generated labels)
        - [x] batch size (int)
        - [x] shuffle (bool)
    - [ ] fit(dataloader)
        - [ ] loop through dataloader
          calculate gradient for each data, update weights, set grads to zero
    - [ ] calculate grad (xb, yb, model)
        - [ ] make preds
        - [ ] calculate loss
        - [ ] backwardpass
    - [ ] model - which makes prediction
        - [ ] we do:
          wb@weights + bias
    - [ ] init params -> function to init anything to random
    - [ ] loss function (todo)
    - [ ] metric function (todo)
    - [ ] optimization function (todo) (optional params)

## Goal 2 -- Train NN in MNIST dataset
todo

## Some key points to remember

### DataLoaders
DataLoaders handle all the necessary work of batching, shuffling, and loading data efficiently to feed your model during training

### Model
todo

### Loss Function
todo

### Metric Function
todo

### Optimization Function
todo


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