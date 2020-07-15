"""Time series classifier model."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.transformers.compose import ColumnConcatenator


class TimeSeriesClassifier():
    """Time series classifier."""

    def fit(self, sequences, class_labels):
        """Fit the model.

        This trains the time series classification model on the given dataset
        where `sequences` is the same data structure as is provided to DeepEcho
        models and `class_labels` is a list of strings. Note that the time
        series classification model does not take into account the context
        component of the sequence data structure.

        Args:
            sequences (list):
                A list of sequences where each sequence is a dictionary
                containing a data and context field. The data field contains a
                list of lists with the time series.
            class_labels (list):
                A list of strings containing the target class.
        """
        steps = [
            ('concatenate', ColumnConcatenator()),
            ('classify', TimeSeriesForestClassifier(n_estimators=100))
        ]
        self.clf = Pipeline(steps)
        X = self._convert(sequences)
        self.clf.fit(X, class_labels)

    def score(self, sequences, class_labels):
        """Evaluate the model.

        Args:
            sequences (list):
                A list of sequences where each sequence is a dictionary
                containing a data and context field. The data field contains a
                list of lists with the time series.
            class_labels (list):
                A list of strings containing the target class.
        """
        X = self._convert(sequences)
        return self.clf.score(X, class_labels)

    def _convert(self, sequences):
        max_seq_len = 0
        for sequence in sequences:
            max_seq_len = max(max_seq_len, len(sequence['data'][0]))

        rows = []
        for sequence in sequences:
            row = {}
            for i, channel in enumerate(sequence['data']):
                channel = [0] * (max_seq_len - len(channel)) + channel
                row['dim_{}'.format(i)] = np.array([(c if c else 0.0) for c in channel])

            rows.append(row)

        return pd.DataFrame(rows)
