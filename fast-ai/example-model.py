"""
    Copy of the simple linear model and neural net trained in
    https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fastai.data.transforms import RandomSplitter
from torch import tensor

TRAIN_DATA = "resources/titanic/train.csv"


def calc_preds(coeffs, indeps):
    return torch.sigmoid(indeps @ coeffs)


def calc_loss(coeffs, indeps, dep, pred_fn):
    return torch.abs(pred_fn(coeffs, indeps) - dep).mean()


def calc_acc(coeffs, val_indep, val_dep, pred_fn=calc_preds):
    return (val_dep.bool() == (pred_fn(coeffs, val_indep) > 0.5)).float().mean()


def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()


def one_epoch(coeffs, lr, trn_indep, trn_dep, pred_fn, update_fn):
    loss = calc_loss(coeffs, trn_indep, trn_dep, pred_fn)
    loss.backward()
    with torch.no_grad():
        update_fn(coeffs, lr)
    print(f"{loss:.3f}", end="; ")


def init_coeffs(n):
    return (torch.randn(n, 1) - 0.5).requires_grad_()


def train_model(
    trn_indep,
    trn_dep,
    epochs=30,
    lr=0.1,
    init_fn=init_coeffs,
    pred_fn=calc_preds,
    update_fn=update_coeffs,
):
    torch.manual_seed(442)
    coeffs = init_fn(trn_indep.shape[1])
    for _ in range(epochs):
        one_epoch(coeffs, lr, trn_indep, trn_dep, pred_fn, update_fn)
    return coeffs


def init_nn_coeffs(n, n_hidden=20):
    layer1 = (torch.rand(n, n_hidden) - 0.5) / n_hidden
    layer2 = torch.rand(n_hidden, 1) - 0.3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(), layer2.requires_grad_(), const.requires_grad_()


def update_nn_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()


def calc_nn_preds(coeffs, indeps):
    l1, l2, c = coeffs
    res = F.relu(indeps @ l1)
    res = res @ l2 + c
    return torch.sigmoid(res)


def init_deep_nn_coeffs(n):
    hiddens = [10, 10]
    sizes = [n] + hiddens + [1]
    n_layers = len(sizes)
    layers = [
        (torch.rand(sizes[i], sizes[i + 1]) - 0.3) / sizes[i + 1] * 4
        for i in range(n_layers - 1)
    ]
    consts = [(torch.rand(1)[0] - 0.5) * 0.1 for i in range(n_layers - 1)]
    for l in layers + consts:
        l.requires_grad_()
    return layers, consts


def calc_deep_nn_preds(coeffs, indeps):
    layers, consts = coeffs
    n = len(layers)
    res = indeps
    for i, l in enumerate(layers):
        res = res @ l + consts[i]
        # only apply relu activation to hidden layers
        if i != n - 1:
            res = F.relu(res)
    return torch.sigmoid(res)


def update_deep_nn_coeffs(coeffs, lr):
    layers, consts = coeffs
    for layer in layers + consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()


def _main():
    ### Linear model ###

    df = pd.read_csv(TRAIN_DATA)
    print(df)

    print(df.isna().sum())

    modes = df.mode().iloc[0]
    df.fillna(modes, inplace=True)

    df["LogFare"] = np.log(df["Fare"] + 1)
    df = pd.get_dummies(df, columns=["Sex", "Pclass", "Embarked"], dtype=float)
    print(df.columns)

    added_cols = [
        "Sex_male",
        "Sex_female",
        "Pclass_1",
        "Pclass_2",
        "Pclass_3",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
    ]
    print(df[added_cols])

    t_dep = tensor(df["Survived"])

    indep_cols = ["Age", "SibSp", "Parch", "LogFare"] + added_cols
    t_indep = tensor(df[indep_cols].values, dtype=torch.float)

    torch.manual_seed(442)

    n_coeff = t_indep.shape[1]
    coeffs = torch.randn(n_coeff) - 0.5

    vals, indices = t_indep.max(dim=0)
    # uses broadcasting to divide the vector into each row of the t_indep matrix
    t_indep = t_indep / vals

    preds = (t_indep * coeffs).sum(axis=1)
    print(preds[:10])

    loss = torch.abs(preds - t_dep).mean()
    print(loss)

    # run one manual epoch of gradient descent

    # methods in pytorch ending in _ are in-place
    coeffs.requires_grad_()

    loss = calc_loss(coeffs, t_indep, t_dep, calc_preds)
    loss.backward()
    with torch.no_grad():
        coeffs.sub_(coeffs.grad * 0.1)
        coeffs.grad.zero_()
        print(calc_loss(coeffs, t_indep, t_dep, calc_preds))

    trn_split, val_split = RandomSplitter(seed=42)(df)

    trn_indep, val_indep = t_indep[trn_split], t_indep[val_split]
    trn_dep, val_dep = t_dep[trn_split], t_dep[val_split]
    print((len(trn_indep), len(val_indep)))

    # transform into column vectors
    trn_dep = trn_dep[:, None]
    val_dep = val_dep[:, None]

    print("\nTranining linear model...")
    coeffs = train_model(trn_indep, trn_dep, epochs=18, lr=0.1)
    print()
    print(coeffs)
    print(calc_acc(coeffs, val_indep, val_dep))

    coeffs = train_model(trn_indep, trn_dep, lr=100)
    print(calc_acc(coeffs, val_indep, val_dep))

    ### Neural net ###
    print("\n\nTraining neural net...")
    nn_coeffs = train_model(
        trn_indep,
        trn_dep,
        lr=1.4,
        init_fn=init_nn_coeffs,
        pred_fn=calc_nn_preds,
        update_fn=update_nn_coeffs,
    )
    print()
    print(calc_acc(nn_coeffs, val_indep, val_dep, pred_fn=calc_nn_preds))

    ### Deep neural net ###
    print("\n\nTraining deep neural net...")
    deep_nn_coeffs = train_model(
        trn_indep,
        trn_dep,
        lr=4,
        init_fn=init_deep_nn_coeffs,
        pred_fn=calc_deep_nn_preds,
        update_fn=update_deep_nn_coeffs,
    )
    print()
    print(calc_acc(deep_nn_coeffs, val_indep, val_dep, pred_fn=calc_deep_nn_preds))


if __name__ == "__main__":
    _main()
