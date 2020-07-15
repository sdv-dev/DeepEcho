"""Time Series Classification task."""

import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from deepecho.benchmark.classifier import TimeSeriesClassifier
from deepecho.benchmark.tasks.base import Task


class ClassificationTask(Task):
    """Time series classification tasks."""

    def __init__(self, path_to_task):
        self.path_to_task = path_to_task
        with open(os.path.join(path_to_task, 'task.json'), 'rt') as fin:
            task = json.load(fin)

        self.df = self._load_dataframe()
        self.df = self.df.drop(task['ignored'], axis=1)
        self.key = task['key']
        self.context = [task['target']]

    def evaluate(self, model):
        """Evaluate the given model."""
        sequences, context_types, data_types = self._as_sequences()
        model.fit_sequences(sequences, context_types, data_types)

        synthetic_sequences = []
        for seq in tqdm(sequences, 'Sampling'):
            synthetic_sequences.append({
                'context': seq['context'],
                'data': model.sample_sequence(seq['context'])
            })

        train_sequences, test_sequences = train_test_split(sequences)

        clf = TimeSeriesClassifier()
        clf.fit(train_sequences, np.array([s['context'][0] for s in train_sequences]))
        real_acc = clf.score(test_sequences, np.array([s['context'][0] for s in test_sequences]))

        clf = TimeSeriesClassifier()
        clf.fit(synthetic_sequences, np.array([s['context'][0] for s in synthetic_sequences]))
        synthetic_acc = clf.score(
            test_sequences,
            np.array([s['context'][0] for s in test_sequences])
        )

        report = self._report(sequences, synthetic_sequences)
        return {
            'dataset': self.path_to_task,
            'sdmetrics': report.overall(),
            'real_acc': real_acc,
            'synthetic_acc': synthetic_acc,
        }
