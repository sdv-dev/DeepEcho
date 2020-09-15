"""Dummy models used as baselines for the benchmarking."""

import numpy as np
import pandas as pd

from deepecho.models.base import DeepEcho


class RealDataBaseline(DeepEcho):   # pylint: disable=W0223
    """Dummy model that always returns the real data seen during training.

    This model ignores all the inputs and always returns the dataframe
    that it was given during training, independently on the amount
    of entities requested or the context values passed.

    This is used during the benchmarking to emulate a "perfect" model.
    """

    _data = None

    def fit(self, data, entity_columns=None, context_columns=None,
            data_types=None, segment_size=None, sequence_index=None):
        """Store the data."""
        del entity_columns, context_columns, data_types, segment_size, sequence_index
        self._data = data

    def sample(self, num_entities=None, context=None, sequence_length=None):
        """Return a copy of the stored data."""
        del num_entities, context, sequence_length
        return self._data.copy()


class UniformBaseline(DeepEcho):
    """Model that samples each variable using a uniform distribution.

    This model learns a few simple properties from the input data:
        - The average sequence length
        - The list of unique values in the categorical variables
        - The min and max values in the numerical variables
        - Whether each numerical variable is integer or float

    Then, when sampling, it generates it variable independently using
    a uniform distribution with the learned min/max, choices and type
    properties.

    This model exists as a lower bound, purely random baseline.
    """

    _sequence_length = None
    _data_stats = None

    def fit_sequences(self, sequences, context_types, data_types):
        """Learn basic properties from the data."""
        del context_types
        self._sequence_length = int(np.mean([
            len(sequence['data'][0])
            for sequence in sequences
        ]))

        data = pd.DataFrame()
        for sequence in sequences:
            sequence_df = pd.DataFrame(dict(enumerate(sequence['data'])))
            data = data.append(sequence_df, ignore_index=True)

        data_stats = []
        for idx, data_type in enumerate(data_types):
            column = data[idx]
            if data_type == 'categorical':
                data_stats.append({
                    'type': 'categorical',
                    'choices': column.unique(),
                })
            else:
                data_stats.append({
                    'type': 'integer' if data_type in ['ordinal', 'count'] else 'float',
                    'min': column.min(),
                    'max': column.max(),
                })

        self._data_stats = data_stats

    def sample_sequence(self, context, sequence_length=None):
        """Sample a sequence using a uniform distribution for each variable."""
        del context, sequence_length
        sequence = []
        for stats in self._data_stats:
            data_type = stats['type']
            if data_type == 'categorical':
                choices = stats['choices']
                values = np.random.choice(choices, self._sequence_length)
            elif data_type == 'integer':
                values = np.random.randint(stats['min'], stats['max'], self._sequence_length)
            elif data_type == 'float':
                values = np.random.uniform(stats['min'], stats['max'], self._sequence_length)

            sequence.append(values.tolist())

        return sequence
