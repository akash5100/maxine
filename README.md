# Only help
- Calculate Gradient from pytorch

## Goal 1 -- Implement `Learner` from scratch

Here is my idea, top of the list represents class/function and below it represents its parameters. 

- Learner
    - data loader
        - collection (iter)
            - x_train (actual data)
            - y_trian (generated labels)
        - batch size (int)
        - shuffle (bool)
    - model (todo)
    - loss function (todo)
    - metric function (todo)
    - optimization function (todo) (optional params)

## Goal 2 -- Train NN in MNIST dataset
todo

```bash
pip install -r requirements.txt
```

Run unittest:
```bash
python -m unittest discover
```