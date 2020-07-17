"""Time Series Classification based metrics."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.transformers.compose import ColumnConcatenator


def _build_xy(data, entity_columns, target_column):
    X = pd.DataFrame()
    y = pd.Series()
    for entity_id, df in data.groupby(entity_columns):
        y = y.append(pd.Series({entity_id: df.pop(target_column).iloc[0]}))
        x = df.drop(entity_columns, axis=1)
        x = pd.Series({
            c: x[c].fillna(x[c].mean()).values
            for c in x.columns
        }, name=entity_id)
        X = X.append(x)

    return X, y


def _build_x(data, entity_columns, context_columns):
    X = pd.DataFrame()
    for entity_id, df in data.groupby(entity_columns):
        x = df.drop(entity_columns + context_columns, axis=1)
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
    real_X, real_y = _build_xy(dataset.data, dataset.entity_columns, target_column)
    synt_X, _ = _build_xy(synthetic, dataset.entity_columns, target_column)

    train_index, test_index = train_test_split(real_X.index)
    real_X_train, real_X_test = real_X.loc[train_index], real_X.loc[test_index]
    real_y_train, real_y_test = real_y.loc[train_index], real_y.loc[test_index]
    synt_X_train, synt_X_test = synt_X.loc[train_index], synt_X.loc[test_index]

    real_acc = _score_classifier(real_X_train, real_X_test, real_y_train, real_y_test)
    synt_acc = _score_classifier(synt_X_train, synt_X_test, real_y_train, real_y_test)

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
    real_X = _build_x(dataset.data, dataset.entity_columns, dataset.context_columns)
    synt_X = _build_x(synthetic, dataset.entity_columns, dataset.context_columns)

    X = pd.concat([real_X, synt_X])
    y = np.array([0] * len(real_X) + [1] * len(synt_X))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    detection_score = _score_classifier(X_train, X_test, y_train, y_test)

    return 1 - detection_score
