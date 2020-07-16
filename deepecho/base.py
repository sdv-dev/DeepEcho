"""Base DeepEcho class."""

import pandas as pd
from tqdm import tqdm


class DeepEcho():
    """The base class for DeepEcho models."""

    verbose = True

    def _assemble(self, data):
        sequences = []
        context_types = ['categorical'] * len(self.context_columns)
        data_types = None

        for _, group in data.groupby(self.entity_columns):
            sequence = {}
            group = group.drop(self.entity_columns, axis=1)

            sequence['context'] = group[self.context_columns].iloc[0].tolist()
            group = group.drop(self.context_columns, axis=1)

            sequence['data'] = []
            self.data_columns = list(group.columns)
            for column in group.columns:
                sequence['data'].append(group[column].values.tolist())

            data_types = []
            for column in group.columns:
                dtype = group[column].dtype
                kind = dtype.kind
                if kind in 'fiu':
                    data_types.append('continuous')
                elif kind in 'OSU':
                    data_types.append('categorical')
                else:
                    raise ValueError('Unknown type: {}'.format(dtype))

            sequences.append(sequence)

        return sequences, context_types, data_types

    @staticmethod
    def validate(sequences, context_types, data_types):
        """Validate the model input.

        Args:
            sequences:
                See `fit`.
            context_types:
                See `fit`.
            data_types:
                See `fit`.
        """
        DTYPES = set(['continuous', 'categorical', 'ordinal', 'count', 'datetime'])
        assert all(dtype in DTYPES for dtype in context_types)
        assert all(dtype in DTYPES for dtype in data_types)

        for sequence in sequences:
            assert len(sequence['context']) == len(context_types)
            assert len(sequence['data']) == len(data_types)
            lengths = [len(x) for x in sequence['data']]
            assert len(set(lengths)) == 1

    def fit_sequences(self, sequences, context_types, data_types):
        """Fit a model to the specified sequences.

        Args:
            sequences:
                List of sequences. Each sequence is a single training example
                (i.e. an example of a multivariate time series with some context).
                For example, a sequence might look something like::

                    {
                        "context": [1],
                        "data": [
                            [1, 3, 4, 5, 11, 3, 4],
                            [2, 2, 3, 4,  5, 1, 2],
                            [1, 3, 4, 5,  2, 3, 1]
                        ]
                    }

                The "context" attribute maps to a list of variables which
                should be used for conditioning. These are variables which
                do not change over time.

                The "data" attribute contains a list of lists corrsponding
                to the actual time series data such that `data[i][j]` contains
                the value at the jth time step of the ith channel of the
                multivariate time series.
            context_types:
                List of strings indicating the type of each value in context.
                he value at `context[i]` must match the type specified by
                `context_types[i]`. Valid types include the following: `categorical`,
                `continuous`, `ordinal`, `count`, and `datetime`.
            data_types:
                List of strings indicating the type of each channel in data.
                Each value in the list at data[i] must match the type specified by
                `data_types[i]`. The valid types are the same as for `context_types`.
        """
        raise NotImplementedError()

    def fit(self, data, entity_columns, context_columns=None, data_types=None):
        """Fit the model to a dataframe containing time series data.

        Args:
            data (pd.DataFrame):
                DataFrame containing all the timeseries data alongside the
                entity and context columns.
            entity_columns (list[str]):
                Names of the columns which identify different time series
                sequences. These will be used to group the data in separated
                training examples.
            context_columns (list[str]):
                The columns in the dataframe which are constant within each
                group/entity. These columns will be provided at sampling time
                (i.e. the samples will be conditioned on the context variables).
            data_types (dict[str, str]):
                Dictinary indicating the data types of each column.
        """
        self.entity_columns = entity_columns
        self.context_columns = context_columns or []

        # Convert to sequences
        sequences, _ctypes, _dtypes = self._assemble(data)
        if data_types:
            context_types = [data_types[c] for c in self.context_columns]
            data_types = [data_types[c] for c in self.data_columns]
        else:
            context_types = _ctypes
            data_types = _dtypes

        # Validate and fit
        self.validate(sequences, context_types, data_types)
        self.fit_sequences(sequences, context_types, data_types)

        # Store context values
        self._context = data[self.context_columns]

    def sample_sequence(self, context):
        """Sample a single sequence conditioned on context.

        Args:
            context (list):
                The list of values to condition on. It must match
                the types specified in context_types when fit was called.

        Returns:
            list[list]:
                A list of lists (data) corresponding to the types specified
                in data_types when fit was called.
        """
        raise NotImplementedError()

    def sample(self, num_entities=None, context=None):
        """Sample a dataframe containing time series data.

        Args:
            num_entities (int):
                The number of entities to sample.
            context (pd.DataFrame):
                Context values to use when sampling.

        Returns:
            pd.DataFrame:
                A DataFrame which resembles the original dataframe where (1) the
                entity column(s) are arbitrarily generated, (2) the context
                column(s) are resampled from the original data, and (3) the data
                columns containing the time series comes from the conditional
                time series model.
        """
        columns = self.entity_columns + self.context_columns + self.data_columns

        if context is None:
            if num_entities is None:
                raise TypeError('Either context or num_entities must be not None')

            context = self._context.sample(num_entities, replace=True)
            context = context.reset_index(drop=True)

        else:
            num_entities = len(context)
            context = context.copy()

        for column in self.entity_columns:
            if column not in context:
                context[column] = range(num_entities)

        # Set the entity_columns as index to properly iterate over them
        context = context.set_index(self.entity_columns)

        if self.verbose:
            iterator = tqdm(context.iterrows(), total=num_entities)
        else:
            iterator = context.iterrows()

        groups = pd.DataFrame()
        for entity_values, context_values in iterator:
            context_values = context_values.tolist()
            sequence = self.sample_sequence(context_values)

            # Reformat as a DataFrame
            group = pd.DataFrame(dict(zip(self.data_columns, sequence)), columns=columns)
            group[self.entity_columns] = entity_values
            group[self.context_columns] = context_values

            groups = groups.append(group)

        return groups
