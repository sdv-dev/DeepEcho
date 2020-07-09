from random import randrange

import pandas as pd


class DeepEcho():
    """The base class for DeepEcho models.
    """

    def fit(self, df, entity_columns, context_columns=[], dtypes=None):
        """Fit the model to a dataframe containing time series data.

        Args:
            df: The number of entities.

            entity_columns: The columns in the dataframe to group by in order
                to obtain separate training examples.

            context_columns: The columns in the dataframe which are constant
                within each group/entity. These columns will be provided at
                sampling time (i.e. the samples will be conditioned on the
                context variables).

            context_types: Data types for the context columns; see `fit_sequences`.

            data_types: Data types for the data columns; see `fit_sequences`.
        """
        self.entity_columns = entity_columns
        self.context_columns = context_columns

        # Convert to sequences
        sequences, _ctypes, _dtypes = self._assemble(df, entity_columns, context_columns)
        if dtypes:
            context_types = [dtypes[c] for c in self.context_columns]
            data_types = [dtypes[c] for c in self.data_columns]
        else:
            context_types = _ctypes
            data_types = _dtypes

        # Validate and fit
        self.validate(sequences, context_types, data_types)
        self.fit_sequences(sequences, context_types, data_types)

        # Store context values
        self._context = df[self.context_columns]

    def sample(self, nb_entities):
        """Sample a dataframe containing time series data.

        Args:
            nb_entities: The number of entities.

        Return:
            A DataFrame which resembles the original dataframe where (1) the
            entity column(s) are arbitrarily generated, (2) the context
            column(s) are resampled from the original data, and (3) the data
            columns containing the time series comes from the conditional
            time series model.
        """
        rows = []

        for entity_id in range(nb_entities):
            # Sample data, given resampled context
            context = self._context.iloc[randrange(0, len(self._context))].tolist()
            data = self.sample_sequence(context)

            # Reassemble into dataframe
            for i in range(len(data[0])):
                row = {}
                for _, col_name in enumerate(self.entity_columns):
                    row[col_name] = entity_id
                for j, col_name in enumerate(self.context_columns):
                    row[col_name] = context[j]
                for j, col_name in enumerate(self.data_columns):
                    row[col_name] = data[j][i]
                rows.append(row)

        columns = self.entity_columns + self.context_columns + self.data_columns
        return pd.DataFrame(rows, columns=columns)

    def _assemble(self, df, entity_columns, context_columns):
        sequences = []
        context_types = ["categorical"] * len(context_columns)
        data_types = None

        for _, sub_df in df.groupby(entity_columns):
            sequence = {}
            sub_df = sub_df.drop(entity_columns, axis=1)

            sequence["context"] = sub_df[context_columns].iloc[0].tolist()
            sub_df = sub_df.drop(context_columns, axis=1)

            sequence["data"] = []
            self.data_columns = list(sub_df.columns)
            for column in sub_df.columns:
                sequence["data"].append(sub_df[column].values.tolist())

            data_types = []
            for column in sub_df.columns:
                if sub_df[column].dtype.kind in 'fiu':
                    data_types.append("continuous")
                elif sub_df[column].dtype.kind in 'OSU':
                    data_types.append("categorical")
                else:
                    raise ValueError("Unknown type: %s" % sub_df[column].dtype)

            sequences.append(sequence)

        return sequences, context_types, data_types

    def fit_sequences(self, sequences, context_types, data_types):
        """Fit a model to the specified sequences.

        Args:

            sequences: This is a list of sequences. Each sequence is a
                single training example (i.e. an example of a multivariate
                time series with some context). For example, a sequence
                might look something like::

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

            context_types: This is a list of strings indicating the type
                of each value in context. The value at `context[i]` must match
                the type specified by `context_types[i]`. Valid types include
                the following: `categorical`, `continuous`, `ordinal`,
                `count`, and `datetime`.

            data_types: This is a list of strings indicating the type
                of each channel in data. Each value in the list at data[i]
                must match the type specified by `data_types[i]`. The valid
                types are the same as for `context_types`.
        """
        raise NotImplementedError()

    @staticmethod
    def validate(sequences, context_types, data_types):
        """Validate the model input.

        Args:
            sequences: See `fit`.
            context_types: See `fit`.
            data_types: See `fit`.
        """
        DTYPES = set(["continuous", "categorical", "ordinal", "count", "datetime"])
        assert all(dtype in DTYPES for dtype in context_types)
        assert all(dtype in DTYPES for dtype in data_types)

        for sequence in sequences:
            assert len(sequence["context"]) == len(context_types)
            assert len(sequence["data"]) == len(data_types)
            lengths = [len(x) for x in sequence["data"]]
            assert len(set(lengths)) == 1

    def sample_sequence(self, context):
        """Sample a single sequence conditioned on context.

        Args:
            context: The list of values to condition on. It must match
                the types specified in context_types when fit was called.

        Return:
            A list of lists (data) corresponding to the types specified
            in data_types when fit was called.
        """
        raise NotImplementedError()
