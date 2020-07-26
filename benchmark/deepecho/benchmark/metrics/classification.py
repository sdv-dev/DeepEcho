"""Time Series Classification based metrics."""

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.nn.utils.rnn import pack_sequence

warnings.filterwarnings('ignore')  # pylint: disable=C0413

from sktime.classifiers.compose import TimeSeriesForestClassifier  # noqa isort:skip
from sktime.transformers.compose import ColumnConcatenator  # noqa isort:skip


def _build_xy(data, entity_columns, target_column):
    X = pd.DataFrame()
    y = pd.Series()
    for entity_id, group in data.groupby(entity_columns):
        y = y.append(pd.Series({entity_id: group.pop(target_column).iloc[0]}))
        x = group.drop(entity_columns, axis=1)
        x = pd.Series({
            c: x[c].fillna(x[c].mean()).values
            for c in x.columns
        }, name=entity_id)
        X = X.append(x)

    return X, y


def _build_x(data, entity_columns, context_columns):
    X = pd.DataFrame()
    for entity_id, group in data.groupby(entity_columns):
        x = group.drop(entity_columns + context_columns, axis=1)
        x = pd.Series({
            c: x[c].fillna(x[c].mean()).values
            for c in x.columns
        }, name=entity_id)
        X = X.append(x)

    return X


def _score_classifier(X_train, X_test, y_train, y_test):
    steps = [
        ('concatenate', ColumnConcatenator()),
        ('classify', TimeSeriesForestClassifier(n_estimators=100))
    ]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def _x_to_packed_sequence(X):
    sequences = []
    for _, row in X.iterrows():
        sequence = []
        for _, values in row.iteritems():
            sequence.append(values)
        sequences.append(torch.FloatTensor(sequence).T)
    return pack_sequence(sequences)


def _lstm_score_classifier(X_train, X_test, y_train, y_test):
    input_dim = len(X_train.columns)
    output_dim = len(set(y_train))
    hidden_dim = 32

    lstm = torch.nn.LSTM(input_dim, hidden_dim)
    linear = torch.nn.Linear(hidden_dim, output_dim)
    X_train, X_test = map(_x_to_packed_sequence, (X_train, X_test))
    y_train, y_test = map(torch.LongTensor, (y_train, y_test))

    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(linear.parameters()), lr=1e-2)
    for _ in range(1024):
        _, (y, _) = lstm(X_train)
        y_pred = linear(y[0])
        loss = torch.nn.functional.cross_entropy(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, (y, _) = lstm(X_test)
    y_pred = linear(y[0])
    return torch.sum(y_test == torch.argmax(y_pred, axis=1)).item() / len(y_test)


def classification_score(dataset, synthetic):
    """Compare the performance of a classifier on real data vs synthetic.

    The score is computed by fitting two TimeSeriesForestClassifier instances
    on the real and the synthetic data respectively and then evaluating the
    performance of both instances on held out partition of the real data.

    The obtained accuracy scores are divided (synthetic accuracy / real accuracy)
    to obtain a ratio of how close the synthetic is accuracy is to the real one.

    Args:
        dataset (Dataset):
            The real dataset.
        synthetic (DataFrame):
            The sampled data.

    Returns:
        float
    """
    context_columns = dataset.context_columns
    if len(context_columns) > 1:
        raise ValueError('This metric only works for datasets with single column context')

    target_column = context_columns[0]
    real_x, real_y = _build_xy(dataset.data, dataset.entity_columns, target_column)
    synt_x, _ = _build_xy(synthetic, dataset.entity_columns, target_column)

    train_index, test_index = train_test_split(real_x.index)
    real_x_train, real_x_test = real_x.loc[train_index], real_x.loc[test_index]
    real_y_train, real_y_test = real_y.loc[train_index], real_y.loc[test_index]
    synt_x_train, synt_x_test = synt_x.loc[train_index], synt_x.loc[test_index]

    real_acc = _score_classifier(real_x_train, real_x_test, real_y_train, real_y_test)
    synt_acc = _score_classifier(synt_x_train, synt_x_test, real_y_train, real_y_test)

    return synt_acc / real_acc


def detection_score(dataset, synthetic):
    """Try to detect whether a sequence is synthetic or not using a classifier.

    The detection is performed by fitting a TimeSeriesForestClassifier
    with both real and synthetic senquences and then calculating a score
    over more sequences.

    The obtained score is returned inverted (1 - classification_score) to
    ensure that the metric is increasing.

    Args:
        dataset (Dataset):
            The real dataset.
        synthetic (DataFrame):
            The sampled data.

    Returns:
        float
    """
    real_x = _build_x(dataset.data, dataset.entity_columns, dataset.context_columns)
    synt_x = _build_x(synthetic, dataset.entity_columns, dataset.context_columns)

    X = pd.concat([real_x, synt_x])
    y = np.array([0] * len(real_x) + [1] * len(synt_x))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rf_score = _score_classifier(X_train, X_test, y_train, y_test)
    lstm_score = _lstm_score_classifier(X_train, X_test, y_train, y_test)

    return {
        "rf": 1 - rf_score,
        "lstm": 1 - lstm_score,
    }
