import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from deepecho_benchmark.classifier import TimeSeriesClassifier
from deepecho_benchmark.tasks.base import Task


class SimpleTask(Task):
    """Time series modeling tasks.
    """

    def __init__(self, path_to_task):
        self.path_to_task = path_to_task
        with open(os.path.join(path_to_task, "task.json"), "rt") as fin:
            task = json.load(fin)

        self.df = self._load_dataframe()
        self.df = self.df.drop(task["ignored"], axis=1)
        self.key = task["key"]
        self.context = task["context"]

    def evaluate(self, model):
        sequences, context_types, data_types = self._as_sequences()
        model.fit_sequences(sequences, context_types, data_types)

        synthetic_sequences = []
        for seq in tqdm(sequences, "Sampling"):
            synthetic_sequences.append({
                "context": seq["context"],
                "data": model.sample_sequence(seq["context"])
            })

        X = sequences + synthetic_sequences
        y = np.array([0] * len(sequences) + [1] * len(synthetic_sequences))

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf = TimeSeriesClassifier()
        clf.fit(X_train, y_train)
        detection_acc = clf.score(X_test, y_test)

        report = self._report(sequences, synthetic_sequences)
        return {
            "dataset": self.path_to_task,
            "sdmetrics": report.overall(),
            "detection_acc": detection_acc,
        }
