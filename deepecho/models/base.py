"""Base DeepEcho class."""

import pandas as pd
import tqdm

from deepecho.sequences import assemble_sequences


class DeepEcho:
    """The base class for DeepEcho models."""

    _verbose = True
    _output_columns = None
    _data_columns = None
    _entity_columns = None
    _context_columns = None
    _context_values = None

    @staticmethod
    def _validate(sequences, context_types, data_types):
        """Validate the model input.

        Args:
            sequences:
                See `fit`.
            context_types:
                See `fit`.
            data_types:
                See `fit`.
        """
        dtypes = set([
            'continuous',
            'categorical',
            'ordinal',
            'count',
            'datetime',
        ])
        assert all(dtype in dtypes for dtype in context_types)
        assert all(dtype in dtypes for dtype in data_types)

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
                        'context': [1],
                        'data': [
                            [1, 3, 4, 5, 11, 3, 4],
                            [2, 2, 3, 4, 5, 1, 2],
                            [1, 3, 4, 5, 2, 3, 1],
                        ],
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

    @staticmethod
    def _get_data_types(data, data_types, columns):
        """Analyze the data and tell the data type of each column."""
        dtypes_list = []
        data_types = data_types or {}
        for column in columns:
            if column in data_types:
                dtypes_list.append(data_types[column])
            else:
                dtype = data[column].dtype
                kind = dtype.kind
                if kind in 'fiud':
                    dtypes_list.append('continuous')
                elif kind in 'OSUb':
                    dtypes_list.append('categorical')
                elif kind == 'M':
                    dtypes_list.append('datetime')
                else:
                    error = f'Unsupported data_type for column {column}: {dtype}'
                    raise ValueError(error)

        return dtypes_list

    def fit(
        self,
        data,
        entity_columns=None,
        context_columns=None,
        data_types=None,
        segment_size=None,
        sequence_index=None,
    ):
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
                Dictinary indicating the data types of each column, which can be
                ``categorical``, ``continuous`` or ``datetime``.
            segment_size (int, pd.Timedelta or str):
                If specified, cut each training sequence in several segments of the
                indicated size. The size can either can passed as an integer value,
                which will interpreted as the number of data points to put on each
                segment, or as a pd.Timedelta (or equivalent str representation),
                which will be interpreted as the segment length in time. Timedelta
                segment sizes can only be used with sequence indexes of type datetime.
            sequence_index (str):
                Name of the column that acts as the order index of each sequence.
                The sequence index column can be of any type that can be sorted,
                such as integer values or datetimes.
        """
        if not entity_columns and segment_size is None:
            raise TypeError('If the data has no `entity_columns`, `segment_size` must be given.')
        if segment_size is not None and not isinstance(segment_size, int):
            if sequence_index is None:
                raise TypeError(
                    '`segment_size` must be of type `int` if no `sequence_index` is given.'
                )
            if data[sequence_index].dtype.kind != 'M':
                raise TypeError(
                    '`segment_size` must be of type `int` if '
                    '`sequence_index` is not a `datetime` column.'
                )

            segment_size = pd.to_timedelta(segment_size)

        self._output_columns = list(data.columns)
        self._entity_columns = entity_columns or []
        self._context_columns = context_columns or []
        self._data_columns = [
            column
            for column in data.columns
            if column not in self._entity_columns + self._context_columns
        ]
        if sequence_index:
            self._output_columns.remove(sequence_index)
            self._data_columns.remove(sequence_index)

        data_types = self._get_data_types(data, data_types, self._data_columns)
        context_types = self._get_data_types(data, data_types, self._context_columns)
        sequences = assemble_sequences(
            data,
            self._entity_columns,
            self._context_columns,
            segment_size,
            sequence_index,
        )

        # Validate and fit
        self._validate(sequences, context_types, data_types)
        self.fit_sequences(sequences, context_types, data_types)

        # Store context values
        self._context_values = data[self._context_columns]

    def sample_sequence(self, context, sequence_length=None):
        """Sample a single sequence conditioned on context.

        Args:
            context (list):
                The list of values to condition on. It must match
                the types specified in context_types when fit was called.
            sequence_length (int or None):
                If given, force sequences to be of the indicated length.
                If ``None`` (default), sample sequences of the same length
                as the original dataset.

        Returns:
            list[list]:
                A list of lists (data) corresponding to the types specified
                in data_types when fit was called.
        """
        raise NotImplementedError()

    def sample(self, num_entities=None, context=None, sequence_length=None):
        """Sample a dataframe containing time series data.

        Args:
            num_entities (int):
                The number of entities to sample.
            context (pd.DataFrame):
                Context values to use when sampling.
            sequence_length (int or None):
                If given, force sequences to be of the indicated length.
                If ``None`` (default), sample sequences of the same length
                as the original dataset.

        Returns:
            pd.DataFrame:
                A DataFrame which resembles the original dataframe where (1) the
                entity column(s) are arbitrarily generated, (2) the context
                column(s) are resampled from the original data, and (3) the data
                columns containing the time series comes from the conditional
                time series model.
        """
        if context is None:
            if num_entities is None:
                raise TypeError('Either context or num_entities must be not None')

            context = self._context_values.sample(num_entities, replace=True)
            context = context.reset_index(drop=True)

        else:
            num_entities = len(context)
            context = context.copy()

        for column in self._entity_columns:
            if column not in context:
                context[column] = range(num_entities)

        # Set the entity_columns as index to properly iterate over them
        if self._entity_columns:
            context = context.set_index(self._entity_columns)

        if self._verbose:
            iterator = tqdm.tqdm(context.iterrows(), total=num_entities)
        else:
            iterator = context.iterrows()

        output = pd.DataFrame()
        for entity_values, context_values in iterator:
            context_values = context_values.tolist()
            sequence = self.sample_sequence(context_values, sequence_length)

            # Reformat as a DataFrame
            group = pd.DataFrame(
                dict(zip(self._data_columns, sequence)),
                columns=self._data_columns,
            )
            group[self._entity_columns] = entity_values
            for column, value in zip(self._context_columns, context_values):
                group[column] = value

            output = pd.concat([output, group])

        return output[self._output_columns].reset_index(drop=True)
