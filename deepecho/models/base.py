class BaseModel():
    """The base class for DeepEcho models.
    """

    def fit(self, sequences, context_types, data_types):
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
                to the actual time series data such that data[i][j] contains
                the value at the jth time step of the ith channel of the
                multivariate time series.

            context_types: This is a list of strings indicating the type
                of each value in context. The value at context[i] must match
                the type specified by context_types[i]. Valid types include
                the following: `categorical`, `continuous`, `ordinal`,
                `count`, and `datetime`.

            data_types: This is a list of strings indicating the type
                of each channel in data. Each value in the list at data[i]
                must match the type specified by data_types[i]. The valid
                types are the same as for `context_types`.
        """
        raise NotImplementedError()

    def sample(self, context):
        """Sample a single sequence conditioned on context.

        Args:

            context: The list of values to condition on. It must match
                the types specified in context_types when fit was called.

        Return:
            A list of lists (data) corresponding to the types specified
            in data_types when fit was called.
        """
        raise NotImplementedError()
