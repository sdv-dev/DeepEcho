"""Utils for models."""
# pylint: disable-all

import numpy as np
import pandas as pd
import torch


def context_to_tensor_gan(context, context_size, context_map):
    """Convert the input context to the corresponding tensor."""
    tensor = torch.zeros(context_size)
    for column, properties in context_map.items():
        value = context[column]
        value_to_tensor(tensor, value, properties)

    return tensor


def data_to_tensor_gan(data, model_data_size, data_map, max_sequence_length, fixed_length=None):
    """Convert the input data to the corresponding tensor.

    If ``self._fixed_length`` is ``False``, add a 1.0 to indicate
    the sequence end and pad the rest of the sequence with 0.0s.
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


@staticmethod
def denormalize(tensor, row, properties, round_value, std_norm=False):
    """Denormalize previously normalized values, setting NaN values if necessary.

    If 'std_norm' is True, denormalize from 0 and 1.
    If 'std_norm' is False, denormalize from -1 and 1.
    """
    value_idx, missing_idx = properties['indices']
    if tensor[row, 0, missing_idx] > 0.5:
        return None

    normalized = tensor[row, 0, value_idx].item()
    column_min = properties['min']
    column_range = properties['max'] - column_min

    if std_norm:
        denormalized = (normalized) * column_range + column_min
    else:
        denormalized = (normalized + 1) * column_range / 2.0 + column_min

    if round_value:
        denormalized = round(denormalized)

    return denormalized


@staticmethod
def index_map(columns, types):
    """Decide which dimension will store which column information in the tensor.

    The output of this function has two elements:

        - An idx_map, which is a dict that indicates the indexes at which
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


@staticmethod
def normalize(tensor, value, properties, std_norm=False):
    """Normalize value and flag nans.

    If 'std_norm' is True, normalize between 0 and 1.
    If 'std_norm' is False, normalize between -1 and 1.
    """
    value_idx, missing_idx = properties['indices']
    if pd.isnull(value):
        tensor[value_idx] = 0.0
        tensor[missing_idx] = 1.0
    else:
        column_min = properties['min']
        column_range = properties['max'] - column_min
        offset = value - column_min

        if std_norm:
            tensor[value_idx] = offset / column_range
        else:
            tensor[value_idx] = 2.0 * offset / column_range - 1.0

        tensor[missing_idx] = 0.0

    return tensor


@staticmethod
def one_hot_decode(tensor, row, properties):
    """Obtain the category that corresponds to the highest one-hot value."""
    max_value = float('-inf')
    for category, idx in properties['indices'].items():
        value = tensor[row, 0, idx]
        if value > max_value:
            max_value = value
            selected = category

    return selected


@staticmethod
def one_hot_encode(tensor, value, properties):
    """Update the index that corresponds to the value to 1.0."""
    value_index = properties['indices'][value]
    tensor[value_index] = 1.0

    return tensor


def transform(data, data_map):
    """Transform data."""
    for properties in data_map.values():
        column_type = properties['type']
        if column_type in ('continuous', 'count'):
            value_idx, missing_idx = properties['indices']
            data[:, :, value_idx] = torch.tanh(data[:, :, value_idx])
            data[:, :, missing_idx] = torch.sigmoid(data[:, :, missing_idx])
        elif column_type in ('categorical', 'ordinal'):
            indices = list(properties['indices'].values())
            data[:, :, indices] = torch.nn.functional.gumbel_softmax(
                data[:, :, indices], hard=True)

    return data


def truncate(generated, data_size):
    """Truncate generated samples."""
    end_flag = (generated[:, :, data_size] > 0.5).float().round()
    generated[:, :, data_size] = end_flag

    for sequence_idx in range(generated.shape[1]):
        # Pad with zeroes after end_flag == 1
        sequence = generated[:, sequence_idx]
        end_flag = sequence[:, data_size]
        if (end_flag == 1.0).any():
            cut_idx = end_flag.detach().cpu().numpy().argmax()
            sequence[cut_idx + 1:] = 0.0


def tensor_to_data_gan(tensor, data_map):
    """Rebuild a valid sequence from the given tensor."""
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
                raise ValueError()   # Theoretically unreachable

            column_data.append(value)

    return data


def value_to_tensor(tensor, value, properties):
    """Update the tensor according to the value and properties."""
    column_type = properties['type']
    if column_type in ('continuous', 'count'):
        normalize(tensor, value, properties)
    elif column_type in ('categorical', 'ordinal'):
        one_hot_encode(tensor, value, properties)

    else:
        raise ValueError()   # Theoretically unreachable
