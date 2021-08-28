"""Utils for models."""

import numpy as np
import pandas as pd
import torch


def _get_dimension(dim, learn):
    if learn == 'value':
        indices = (dim, dim + 1)
        dim += 2

    elif learn == 'dist':
        indices = (dim, dim + 1, dim + 2)
        dim += 3

    else:
        raise ValueError('Unsupported learning criteria: {}'.format(learn))

    return indices, dim


def index_map(columns, types, transformers, learn='value'):
    """Decide which dimension will store which column information in the tensor.

    The output of this function has two elements:

        - A 'mapping', which is a dict that indicates the indexes at which
          the list of tensor dimensions associated with each input column starts,
          and the properties of such columns.
        - An integer that indicates how many dimensions the tensor will have.

    In order to decide this, the following process is followed for each column:

        - If ``learn`` is set to value, 2 dimensions are created for it. These
          will contain information about the value itself, as well as information
          about whether the value should be NaN or not.
        - If ``learn`` is set to dist, 3 dimensions are created for it. These
          will contain information about the first parameter of a distribution
          (e.g. mu), the second parameter of a distribution (e.g. std), as well
          as information about whether the value should be NaN or not.
        - If the transformation is one hot (i.e. categorical or ordinal columns),
          a single dimension is created for each possible value, which will be
          later on used to hold one-hot encoding information about the values.

    Args:
        columns (list):
            List of lists containing the values of each column.
        types (list):
            List of strings containing the type of each column.
        transormers (dict):
            Dictionary specifying the transformation per column type.
        learn (str):
            String denoting whether the model is learning to estimate
            the value or the parameters of a distribution.

    Returns:
        tuple:
            * ``dict``: Information related to the properties of the columns data.
            * ``int``: Number of dimensions the that tensor will have.
    """
    dimensions = 0
    mapping = {}
    for column, column_type in enumerate(types):
        values = columns[column]
        if transformers[column_type] == 'minmax':
            indices, dimensions = _get_dimension(dimensions, learn)
            mapping[column] = {
                'type': column_type,
                'transform': 'minmax',
                'min': np.nanmin(values),
                'max': np.nanmax(values),
                'nulls': pd.isnull(values).any(),
                'indices': indices
            }

        elif transformers[column_type] == 'zscore':
            indices, dimensions = _get_dimension(dimensions, learn)
            mapping[column] = {
                'type': 'continuous',
                'transform': 'zscore',
                'mu': np.nanmean(values),
                'std': np.nanstd(values),
                'nulls': pd.isnull(values).any(),
                'indices': indices
            }

        elif transformers[column_type] == 'one-hot':
            indices = {}
            for value in pd.unique(values):
                if pd.isnull(value):
                    value = None

                indices[value] = dimensions
                dimensions += 1

            mapping[column] = {
                'type': column_type,
                'transform': 'one-hot',
                'indices': indices
            }

        else:
            raise ValueError('Unsupported type: {}'.format(column_type))

    return mapping, dimensions


def _minmax_scaling(value, properties):
    minimum = properties['min']
    maximum = properties['max']
    value_range = maximum - minimum
    offset = value - minimum
    return offset / value_range

    # return 2.0 * offset / value_range - 1.0


def _minmax_descaling(value, properties):
    minimum = properties['min']
    maximum = properties['max']
    value_range = maximum - minimum
    return value * value_range + minimum

    # return (value + 1) * value_range / 2.0 + minimum


def _zscore_scaling(value, properties):
    mean = properties['mu']
    std = properties['std']

    return (value - mean) / std


def _zscore_descaling(value, properties):
    mean = properties['mu']
    std = properties['std']

    return value * std + mean


def normalize(tensor, values, properties, scaler, seq_len):
    """Normalize value and flag nans.

    Normalized values are between -1 and 1.

    Args:
        tensor (array):
            Tensor in which the normalized values will be stored.
        values (array):
            Values to normalize.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices and the min/max values.
        scaler (function):
            Function that scales the data.
    """
    primary_idx, missing_idx, *secondary_idx = properties['indices']
    if secondary_idx:
        temp = secondary_idx[0]
        secondary_idx = missing_idx
        missing_idx = temp

    tensor[:seq_len, missing_idx] = np.isnan(values).astype(float)
    tensor[:seq_len, primary_idx] = scaler(np.nan_to_num(values, nan=0.), properties)
    if secondary_idx:
        tensor[:seq_len, secondary_idx] = torch.zeros(seq_len)


def denormalize(tensor, properties, function, round_value):
    """Denormalize previously normalized values, setting NaN values if necessary.

    Args:
        tensor (array):
            3D Vector that contains different samples with normalized values
            and records of null values.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices and the min/max values.
        function (Function):
            Function to denormalize values.
        round_value (boolean):
            Whether to round the denormalized value or not.

    Returns:
        float:
            Denormalized value.
    """
    primary_idx, missing_idx, *secondary_idx = properties['indices']
    if secondary_idx:
        temp = secondary_idx[0]
        secondary_idx = missing_idx
        missing_idx = temp

    values = tensor[:, primary_idx]
    missed = tensor[:, missing_idx]

    denormalized = function(values, properties)

    if round_value:
        denormalized = np.around(denormalized).astype(int)

    denormalized = np.where(missed > 0.5, None, denormalized)

    return denormalized


def one_hot_encode(tensor, values, properties, seq_len):
    """Set 1.0 at the tensor index that corresponds to the value.

    Args:
        tensor (array):
            Tensor that will be updated.
        values (array):
            Values that need to be one-hot encoded.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices of the values.
        seq_len(int):
            Length of the sequence.
    """
    for i in range(seq_len):  # remove this loop
        value = values[i]
        if pd.isnull(value):
            value = None
        value_index = properties['indices'][value]
        tensor[i, value_index] = 1.0


def one_hot_decode(tensor, properties, sequence_length):
    """Obtain the category that corresponds to the highest one-hot value.

    Args:
        tensor (array):
            Tensor which contains the one-hot encoded rows.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices of the values.
        sequence_length (int):
            Length of the sequence.

    Returns:
        int:
            Decoded category value.
    """
    selected = [None] * sequence_length
    for i in range(sequence_length):
        max_value = float('-inf')
        for category, idx in properties['indices'].items():
            value = tensor[i, idx]
            if value > max_value:
                max_value = value
                selected[i] = category

    return selected


def value_to_tensor(tensor, value, properties, seq_len):
    """Update the tensor according to the value and properties.

    Args:
        tensor (array):
            Tensor in which the encoded or normalized values will be stored.
        value (float):
            Value to encode or normalize.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices and min/max of the values.
    """
    column_transform = properties['transform']

    if column_transform == 'minmax':
        normalize(tensor, value, properties, _minmax_scaling, seq_len)
    elif column_transform == 'zscore':
        normalize(tensor, value, properties, _zscore_scaling, seq_len)
    elif column_transform == 'one-hot':
        one_hot_encode(tensor, value, properties, seq_len)
    else:
        raise ValueError()   # Theoretically unreachable


def data_to_tensor(data, model_data_size, data_map, fixed_length, max_sequence_length):
    """Convert the input data to the corresponding tensor.

    If ``fixed_length`` is ``False``, add a 1.0 to indicate
    the sequence end and pad the rest of the sequence with 0.0s.

    Args:
        data (list):
            List of lists containing the input sequences.
        model_data_size (int):
            Number of columns to create in the tensor.
        data_map (dict):
            Dictionary with information related to the data variables,
            which must contain the indices and min/max of the values.
        fixed_length (boolean):
            Whether to add an end flag column or not.
        max_sequence_length (int):
            Maximum sequence length.

    Returns:
        torch.Tensor
    """
    num_rows = len(data[0])
    tensor = np.zeros((max_sequence_length, model_data_size))
    for column, properties in data_map.items():
        value = data[column]
        value_to_tensor(tensor, value, properties, num_rows)

    if not fixed_length:
        tensor[num_rows - 1][-1] = 1.0

    return torch.Tensor(tensor)


def context_to_tensor(context, context_size, context_map):
    """Convert the input context to the corresponding tensor.

    Args:
        context (list):
            List containing the context values.
        context_size (int):
            Size of the output tensor.
        context_map (dict):
            Dictionary with information related to the context variables,
            which must contain the indices and min/max of the values.

    Returns:
         torch.Tensor
    """
    tensor = np.zeros((1, context_size))
    for column, properties in context_map.items():
        value = [context[column]]
        value_to_tensor(tensor, value, properties, 1)

    return torch.Tensor(tensor.flatten())


def tensor_to_data(tensor, data_map):
    """Rebuild a valid sequence from the given tensor.

    Args:
        tensor (list):
            Tensor containing the generated data.
        data_map (int):
            Dictionary with information related to the data variables,
            which must contain the indices and min/max of the values.

    Returns:
        list
    """
    sequence_length, num_sequences, _ = tensor.shape
    assert num_sequences == 1

    tensor = tensor.squeeze(1)
    array = tensor.detach().numpy()

    data = [None] * len(data_map)
    for column, properties in data_map.items():
        column_transform = properties['transform']
        column_type = properties['type']
        round_value = column_type == 'count'

        if column_transform == 'minmax':
            data[column] = denormalize(array, properties, _minmax_descaling, round_value)
        elif column_transform == 'zscore':
            data[column] = denormalize(array, properties, _zscore_descaling, round_value)
        elif column_transform == 'one-hot':
            data[column] = one_hot_decode(array, properties, sequence_length)
        else:
            raise ValueError()  # Theoretically unreachable

    return data


def build_tensor(transform, sequences, key, dim, **transform_kwargs):
    """Convert input sequences to tensors.

    Args:
        transform (function):
            Function to apply.
        sequences (dict):
            Dict containing the sequences and the context vectors.
        key (str):
            Key to use when obtaining the data from the sequences dict.
        dim (int)
            Dimension to use when the tensors are stacked. If `None`
            do not stack.
        **transform_kwargs(dict)
            Additional arguments for the ``transform`` function.

    Returns:
        torch tensor
    """
    tensors = []
    for sequence in sequences:
        tensors.append(transform(sequence[key], **transform_kwargs))

    if dim is None:
        return tensors

    return torch.stack(tensors, dim=dim)
