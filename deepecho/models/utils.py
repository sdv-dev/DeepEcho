"""Utils for models."""

import numpy as np
import pandas as pd
import torch


def index_map(columns, types):
    """Decide which dimension will store which column information in the tensor.

    The output of this function has two elements:

        - An 'mapping', which is a dict that indicates the indexes at which
          the list of tensor dimensions associated with each input column starts,
          and the properties of such columns.
        - An integer that indicates how many dimensions the tensor will have.

    In order to decide this, the following process is followed for each column:

        - If the column is numerical (continuous or count), 2 dimensions are created
          for it. These will contain information about the value itself, as well
          as information about whether the value should be NaN or not.
        - If the column is categorical or ordinal, 1 dimentions is created for
          each possible value, which will be later on used to hold one-hot encoding
          information about the values.

    Args:
        columns (list):
            List of lists containing the values of each column.
        types (list):
            List of strings containing the type of each column.

    Returns:
        tuple:
            * ``dict``: Information related to the properties of the columns data.
            * ``int``: Number of dimensions the that tensor will have.
    """
    dimensions = 0
    mapping = {}
    for column, column_type in enumerate(types):
        values = columns[column]
        if column_type in ('continuous', 'count'):
            mapping[column] = {
                'type': column_type,
                'min': np.min(values),
                'max': np.max(values),
                'indices': (dimensions, dimensions + 1)
            }
            dimensions += 2

        elif column_type in ('categorical', 'ordinal'):
            indices = {}
            for value in set(values):
                indices[value] = dimensions
                dimensions += 1

            mapping[column] = {
                'type': column_type,
                'indices': indices
            }

        else:
            raise ValueError('Unsupported type: {}'.format(column_type))

    return mapping, dimensions


def normalize(tensor, value, properties):
    """Normalize value and flag nans.

    Normalized values are between -1 and 1.

    Args:
        tensor (array):
            Tensor in which the normalized values will be stored.
        value (float):
            Value to normalize.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices and the min/max values.
    """
    value_idx, missing_idx = properties['indices']
    if pd.isnull(value):
        tensor[value_idx] = 0.0
        tensor[missing_idx] = 1.0
    else:
        column_min = properties['min']
        column_range = properties['max'] - column_min
        offset = value - column_min

        tensor[value_idx] = 2.0 * offset / column_range - 1.0
        tensor[missing_idx] = 0.0


def denormalize(tensor, row, properties, round_value):
    """Denormalize previously normalized values, setting NaN values if necessary.

    Args:
        tensor (array):
            3D Vector that contains different samples with normalized values
            and records of null values.
        row (int):
            Index of the row that needs to be decoded.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices and the min/max values.
        round_value (boolean):
            Whether to round the denormalized value or not.

    Returns:
        float:
            Denormalized value.
    """
    value_idx, missing_idx = properties['indices']
    if tensor[row, 0, missing_idx] > 0.5:
        return None

    normalized = tensor[row, 0, value_idx].item()
    column_min = properties['min']
    column_range = properties['max'] - column_min

    denormalized = (normalized + 1) * column_range / 2.0 + column_min

    if round_value:
        denormalized = round(denormalized)

    return denormalized


def one_hot_encode(tensor, value, properties):
    """Set 1.0 at the tensor index that corresponds to the value.

    Args:
        tensor (array):
            Tensor that will be updated.
        value (int):
            Value that needs to be one-hot encoded.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices of the values.
    """
    value_index = properties['indices'][value]
    tensor[value_index] = 1.0


def one_hot_decode(tensor, row, properties):
    """Obtain the category that corresponds to the highest one-hot value.

    Args:
        tensor (array):
            Tensor which contains the one-hot encoded rows.
        row (int):
            Index of the row that needs to be decoded.
        properties (dict):
            Dictionary with information related to the given value,
            which must contain the indices of the values.

    Returns:
        int:
            Decoded category value.
    """
    max_value = float('-inf')
    for category, idx in properties['indices'].items():
        value = tensor[row, 0, idx]
        if value > max_value:
            max_value = value
            selected = category

    return selected


def value_to_tensor(tensor, value, properties):
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
    column_type = properties['type']
    if column_type in ('continuous', 'count'):
        normalize(tensor, value, properties)
    elif column_type in ('categorical', 'ordinal'):
        one_hot_encode(tensor, value, properties)

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
        torch.tensor
    """
    tensors = []
    num_rows = len(data[0])
    for row in range(num_rows):
        tensor = torch.zeros(model_data_size)
        for column, properties in data_map.items():
            value = data[column][row]
            value_to_tensor(tensor, value, properties)

        tensors.append(tensor)

    if not fixed_length:
        tensors[-1][-1] = 1.0

    for _ in range(max_sequence_length - num_rows):
        tensors.append(torch.zeros(model_data_size))

    return torch.stack(tensors, dim=0)


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
         torch.tensor
    """
    tensor = torch.zeros(context_size)
    for column, properties in context_map.items():
        value = context[column]
        value_to_tensor(tensor, value, properties)

    return tensor


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

    data = [None] * len(data_map)
    for column, properties in data_map.items():
        column_type = properties['type']

        column_data = []
        data[column] = column_data
        for row in range(sequence_length):
            if column_type in ('continuous', 'count'):
                round_value = column_type == 'count'
                value = denormalize(tensor, row, properties, round_value=round_value)
            elif column_type in ('categorical', 'ordinal'):
                value = one_hot_decode(tensor, row, properties)
            else:
                raise ValueError()  # Theoretically unreachable

            column_data.append(value)

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
            Dimension to use when the tensors are stacked.
        **transform_kwargs(dict)
            Additional arguments for the ``transform`` function.

    Returns:
        torch tensor
    """
    tensors = []
    for sequence in sequences:
        tensors.append(transform(sequence[key], **transform_kwargs))

    return torch.stack(tensors, dim=dim)
