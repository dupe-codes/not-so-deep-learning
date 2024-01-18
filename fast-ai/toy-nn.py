"""
    Giga simple neural network with two linear layers and ReLU activation.

    Mirrors the fast.ai lecture 3 Titanic excel spreadsheet example.

    Plans:
        1. Read data from titanic csv using pandas
        2. Transform data into ML friendly formats (e.g. encode categorical data, normalize continuous data)
        3. Define mse loss function
        5. Define simple neural net work one linear layer > ReLU > linear layer
        6. Train the network on the training data
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# import torch
# from torch import Generator, randn, tensor
# from torch.utils.data import DataLoader, RandomSampler, TensorDataset

TRAIN_DATA = 'resources/titanic/train.csv'

RAW_FEATURES = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked',
]

NUM_HIDDEN_UNITS = 100
RANDOM_SEED = 42
BATCH_SIZE = 40
LEARNING_RATE = 0.001
NUM_EPOCHS = 50


def _transform_data(df):
    # Remove rows with any missing data
    # print(df)

    # Prefer brackets over dot notation access (df.Name)
    # print(df["Name"])

    # Print first few rows
    print(df.head())
    # Get summary statistics for data in the dataframe
    print(df.describe())

    print("info")
    print(df.info())

    print("\n\nisna")
    print(df.isna())

    df = df.dropna(subset=RAW_FEATURES + ['Survived'])
    print('Data after dropping rows with missing data')
    print(df.shape)

    y = df['Survived'].values
    X = df[RAW_FEATURES]

    col_transformer = ColumnTransformer([
        ('onehot', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked']),
        ('normalize', MinMaxScaler(), ['Age', 'SibSp', 'Parch', 'Fare']),
    ], remainder='passthrough')

    X = col_transformer.fit_transform(X)
    feature_names = col_transformer.get_feature_names_out(
        input_features=RAW_FEATURES)
    X = pd.DataFrame(X, columns=feature_names)

    return X, y
    # return tensor(X.values).float(), tensor(y)


def _mse_loss(y_pred, y_true):
    y_pred = y_pred.sigmoid()
    return ((y_pred - y_true)**2).mean()


def _binary_cross_entropy_loss(y_pred, y_true):
    y_pred = y_pred.sigmoid()
    return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean()


def _accuracy(y_pred, y_true):
    y_pred = y_pred.sigmoid()
    y_pred = (y_pred > 0.5).float()
    return (y_pred == y_true).float().mean()


def _init_params(shape, std=0.1):
    return (std * randn(shape)).requires_grad_()


def _simple_net(X, weights1, bias1, weights2, bias2):
    layer1 = X @ weights1 + bias1
    activation1 = layer1.max(tensor(0.0))
    layer2 = activation1 @ weights2 + bias2
    return layer2


def _train_epoch(params, lr, train_dl, valid_X, valid_y):
    for X_batch, y_batch in train_dl:
        y_pred = _simple_net(X_batch, *params)
        loss = _mse_loss(y_pred, y_batch)
        #loss = _binary_cross_entropy_loss(y_pred, y_batch)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= p.grad * lr
                p.grad.zero_()

    y_pred_valid = _simple_net(valid_X, *params)

    print('Epoch results:')
    print(f'Validation loss: {_mse_loss(y_pred_valid, valid_y)}')
    print(f'Validation accuracy: {_accuracy(y_pred_valid, valid_y)}')
    print('---------')


def main():
    train_df = pd.read_csv(TRAIN_DATA, index_col='PassengerId')
    X, y = _transform_data(train_df)

    # learning note: jumping between pandas dataframes, numpy ndarrays, and torch tensors
    # is a bit messy/confusing
    # X_train, X_test, y_train, y_test = map(tensor, train_test_split(
        # X, y, test_size=0.2, random_state=RANDOM_SEED))

    # ds_train = TensorDataset(X_train, y_train)
    # sampler = RandomSampler(
        # ds_train, generator=Generator().manual_seed(RANDOM_SEED))
    # train_dl = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler)

    # weights1 = _init_params((len(X_train[0]), NUM_HIDDEN_UNITS))
    # bias1 = _init_params(NUM_HIDDEN_UNITS)
    # weights2 = _init_params((NUM_HIDDEN_UNITS, 1))
    # bias2 = _init_params(1)

    # for _ in range(NUM_EPOCHS):
        # _train_epoch((weights1, bias1, weights2, bias2),
                     # LEARNING_RATE, train_dl, X_test, y_test)


if __name__ == '__main__':
    main()
