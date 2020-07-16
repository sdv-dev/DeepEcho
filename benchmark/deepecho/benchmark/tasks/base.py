"""Tasks Base class."""

import json
import os
from glob import glob

import numpy as np
import pandas as pd
import sdmetrics
from sdv import Metadata


class Task():
    """Base class for tasks.

    Tasks can be loaded from disk by calling `Task.load` which will return
    an instance of the appropriate subclass.
    """

    @classmethod
    def load(cls, path_to_task):
        """Load the task from disk.

        Args:
            path_to_task: The path to the task directory.

        Return:
            A Task instance.
        """
        from deepecho.benchmark.tasks.simple import SimpleTask
        from deepecho.benchmark.tasks.classification import ClassificationTask

        task_types = {
            'simple': SimpleTask,
            'classification': ClassificationTask
        }

        with open(os.path.join(path_to_task, 'task.json'), 'rt') as fin:
            task = json.load(fin)

        task_cls = task_types.get(task['task_type'], None)

        if not task_cls:
            raise ValueError('Unknown task type.')

        return task_cls(path_to_task)

    def evaluate(self, model):
        """Evaluate the model on this task.

        Args:
            model: An instance of a DeepEcho model.

        Return:
            A dictionary mapping metric names to values. The set of metrics
            that are returned is fixed for each subclass (i.e. each subclass
            will always return the same set of metrics).
        """
        raise NotImplementedError()

    def _load_dataframe(self):
        # TODO: handle metadata types, multiple files, etc.
        for path_to_csv in glob(os.path.join(self.path_to_task, '*.csv')):
            return pd.read_csv(path_to_csv)

    def _report(self, sequences, synthetic_sequences):
        real_df = []
        for seq in sequences:
            real_df.append(pd.DataFrame(seq['data']).T)

        real_df = pd.concat(real_df, axis=0)

        synthetic_df = []
        for seq in synthetic_sequences:
            synthetic_df.append(pd.DataFrame(seq['data']).T)

        synthetic_df = pd.concat(synthetic_df, axis=0)
        synthetic_df = synthetic_df.astype(np.float64)

        metadata = Metadata()
        metadata.add_table('data', data=real_df)
        real_tables = {'data': real_df}
        synthetic_tables = {'data': synthetic_df}

        return sdmetrics.evaluate(metadata, real_tables, synthetic_tables)

    def _as_sequences(self):
        sequences = []
        context_types = ['categorical']
        data_types = None

        for _, sub_df in self.df.groupby(self.key):
            sequence = {}
            sub_df = sub_df.drop(self.key, axis=1)

            sequence['context'] = sub_df[self.context].iloc[0].tolist()
            sub_df = sub_df.drop(self.context, axis=1)

            sequence['data'] = []
            for column in sub_df.columns:
                sequence['data'].append(sub_df[column].values.tolist())

            data_types = []
            for column in sub_df.columns:
                if sub_df[column].dtype == np.float64:
                    data_types.append('continuous')
                else:
                    raise ValueError('idk')

            sequences.append(sequence)

        return sequences, context_types, data_types
