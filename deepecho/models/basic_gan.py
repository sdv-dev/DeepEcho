"""BasicGAN Model."""

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from deepecho.models.base import DeepEcho

LOGGER = logging.getLogger(__name__)


def _expand_context(data, context):
    return torch.cat(
        [
            data,
            context.unsqueeze(0).expand(data.shape[0], context.shape[0], context.shape[1]),
        ],
        dim=2,
    )


class BasicGenerator(torch.nn.Module):
    """Generator for the BasicGAN model.

    This generator consist on a RNN layer followed by a Linear layer with
    the following schema:

        - The Generator takes as input a ``sequence_length`` and a ``context`` vector.
        - The ``context`` vector is expanded over the ``sequence_lenght`` and padded with
          ``latent_size`` random noise.
        - RNN takes as input a tensor with shape
          ``(sequence_length, context_length, context_size + latent_size)`` and
          generates an output of shape ``(sequence_length, context_length, hidden_size)``.
        - The RNN output is passed to the Linear layer that outputs a tensor of size
          ``(sequence_length, context_length, output_size)``

    Args:
        context_size (int):
            Size of the contextual arrays.
        latent_size (int):
            Size of the random input vector.
        hidden_size (int):
            Size of the communication between the RNN and the Linear layer.
        data_size (int):
            Size of the output layer.
        device (torch.device):
            Device to which this Module is associated to.
    """

    def __init__(self, context_size, latent_size, hidden_size, data_size, device):
        super().__init__()
        self.latent_size = latent_size
        self.rnn = torch.nn.GRU(context_size + latent_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, data_size)
        self.device = device

    def forward(self, context=None, sequence_length=None):
        """Forward computation.

        Args:
            context (tensor):
                Context values to use for each generated sequence.
            sequence_length (int):
                Amount of data points to generate for each sequence.
        """
        latent = torch.randn(
            size=(sequence_length, context.size(0), self.latent_size),
            device=self.device,
        )
        latent = _expand_context(latent, context)

        rnn_out, _ = self.rnn(latent)
        return self.linear(rnn_out)


class BasicDiscriminator(torch.nn.Module):
    """Discriminator for the BasicGAN model.

    This discriminator consist on a RNN layer followed by a Linear layer with
    the following schema:

        - The Discriminator takes as input a collection of sequences that include
          both the data and the context columns.
        - RNN takes as input a tensor with shape
          ``(sequence_length, number_of_sequences, context_size + data_size)`` and
          generates an output of shape ``(sequence_length, num_sequences, hidden_size)``.
        - The RNN output is passed to the Linear layer that outputs a single value.

    Args:
        context_size (int):
            Number of values in the contextual arrays.
        data_size (int):
            Number of variables in the input data sequences.
        hidden_size (int):
            Size of the communication between the RNN and the Linear layer.
    """

    def __init__(self, context_size, data_size, hidden_size):
        super().__init__()
        self.rnn = torch.nn.GRU(context_size + data_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, sequences):
        """Forward computation.

        Args:
            sequences (tensor):
                Sequences of values with their context.
        """
        rnn_out, _ = self.rnn(sequences)
        return self.linear(rnn_out[-1, :, :])


class BasicGANModel(DeepEcho):
    """Basic GAN model.

    0. Normalize continuous values to [-1.0, 1.0].
    1. Map categorical values to one-hot.
    2. Define a generator that takes context vector -> new sequence.
    3. Transforms
        - apply sigmoid to continuous/count/datetime
        - apply softmax to categorical/ordinal
    4. Define a discriminator that takes sequence + context -> score.

    Args:
        epochs (int):
            Number of training epochs. Defaults to 1024.
        latent_size (int):
            Size of the random vector to use for generation. Defaults to 32.
        hidden_size (int):
            Size of the hidden Linear layer. Defaults to 16.
        gen_lr (float):
            Generator learning rate. Defaults to 1e-3.
        dis_lr (float):
            Discriminator learning rate. Defaults to 1e-3.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        verbose (bool):
            Whether to print progress to console or not.
    """

    _max_sequence_length = None
    _fixed_length = None
    _context_map = None
    _context_size = None
    _data_map = None
    _data_size = None
    _model_data_size = None
    _generator = None

    def __init__(
        self,
        epochs=1024,
        latent_size=32,
        hidden_size=16,
        gen_lr=1e-3,
        dis_lr=1e-3,
        cuda=True,
        verbose=True,
    ):
        self._epochs = epochs
        self._gen_lr = gen_lr
        self._dis_lr = dis_lr
        self._latent_size = latent_size
        self._hidden_size = hidden_size

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._verbose = verbose

        LOGGER.info('%s instance created', self)

    def __repr__(self):
        """Return a representation of the class object."""
        return (
            f'{self.__class__.__name__}(epochs={self._epochs}, latent_size={self._latent_size},'
            f'hidden_size={self._hidden_size}, gen_lr={self._gen_lr}, dis_lr={self._dis_lr},'
            f"cuda='{self._device}', verbose={self._verbose})"
        )

    # ########################### #
    # Preprocessing and preparing #
    # ########################### #

    @staticmethod
    def _index_map(columns, types):
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
                    'indices': (dimensions, dimensions + 1),
                }
                dimensions += 2

            elif column_type in ('categorical', 'ordinal'):
                indices = {}
                for value in set(values):
                    indices[value] = dimensions
                    dimensions += 1

                mapping[column] = {'type': column_type, 'indices': indices}

            else:
                raise ValueError(f'Unsupported type: {column_type}')

        return mapping, dimensions

    def _analyze_data(self, sequences, context_types, data_types):
        """Extract information about the context and data that will be used later.

        The following information is stored:
            - Length of the sequences.
            - Index map and dimensions for the context.
            - Index map and dimensions for the data.
        """
        sequence_lengths = np.array([len(sequence['data'][0]) for sequence in sequences])
        self._max_sequence_length = np.max(sequence_lengths)
        self._fixed_length = (sequence_lengths == self._max_sequence_length).all()

        # Concatenate all the context sequences together
        context = []
        for column in range(len(context_types)):
            context.append([sequence['context'][column] for sequence in sequences])

        self._context_map, self._context_size = self._index_map(context, context_types)

        # Concatenate all the data sequences together
        data = []
        for column in range(len(data_types)):
            data.append(sum([sequence['data'][column] for sequence in sequences], []))

        self._data_map, self._data_size = self._index_map(data, data_types)

        self._model_data_size = self._data_size + int(not self._fixed_length)

    @staticmethod
    def _normalize(tensor, value, properties):
        """Normalize the value between 0 and 1 and flag nans."""
        value_idx, missing_idx = properties['indices']
        if pd.isna(value):
            tensor[value_idx] = 0.0
            tensor[missing_idx] = 1.0
        else:
            column_min = properties['min']
            column_range = properties['max'] - column_min
            offset = value - column_min
            tensor[value_idx] = 2.0 * offset / column_range - 1.0
            tensor[missing_idx] = 0.0

    @staticmethod
    def _denormalize(tensor, row, properties, round_value):
        """Denormalize previously normalized values, setting NaN values if necessary."""
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

    @staticmethod
    def _one_hot_encode(tensor, value, properties):
        """Update the index that corresponds to the value to 1.0."""
        value_index = properties['indices'][value]
        tensor[value_index] = 1.0

    @staticmethod
    def _one_hot_decode(tensor, row, properties):
        """Obtain the category that corresponds to the highest one-hot value."""
        max_value = float('-inf')
        for category, idx in properties['indices'].items():
            value = tensor[row, 0, idx]
            if value > max_value:
                max_value = value
                selected = category

        return selected

    def _value_to_tensor(self, tensor, value, properties):
        """Update the tensor according to the value and properties."""
        column_type = properties['type']
        if column_type in ('continuous', 'count'):
            self._normalize(tensor, value, properties)
        elif column_type in ('categorical', 'ordinal'):
            self._one_hot_encode(tensor, value, properties)

        else:
            raise ValueError()  # Theoretically unreachable

    def _data_to_tensor(self, data):
        """Convert the input data to the corresponding tensor.

        If ``self._fixed_length`` is ``False``, add a 1.0 to indicate
        the sequence end and pad the rest of the sequence with 0.0s.
        """
        tensors = []
        num_rows = len(data[0])
        for row in range(num_rows):
            tensor = torch.zeros(self._model_data_size)
            for column, properties in self._data_map.items():
                value = data[column][row]
                self._value_to_tensor(tensor, value, properties)

            tensors.append(tensor)

        if not self._fixed_length:
            tensors[-1][-1] = 1.0

        for _ in range(self._max_sequence_length - num_rows):
            tensors.append(torch.zeros(self._model_data_size))

        return torch.stack(tensors, dim=0)

    def _context_to_tensor(self, context):
        """Convert the input context to the corresponding tensor."""
        tensor = torch.zeros(self._context_size)
        for column, properties in self._context_map.items():
            value = context[column]
            self._value_to_tensor(tensor, value, properties)

        return tensor

    def _tensor_to_data(self, tensor):
        """Rebuild a valid sequence from the given tensor."""
        sequence_length, num_sequences, _ = tensor.shape
        assert num_sequences == 1

        data = [None] * len(self._data_map)
        for column, properties in self._data_map.items():
            column_type = properties['type']

            column_data = []
            data[column] = column_data
            for row in range(sequence_length):
                if column_type in ('continuous', 'count'):
                    round_value = column_type == 'count'
                    value = self._denormalize(tensor, row, properties, round_value=round_value)
                elif column_type in ('categorical', 'ordinal'):
                    value = self._one_hot_decode(tensor, row, properties)
                else:
                    raise ValueError()  # Theoretically unreachable

                column_data.append(value)

        return data

    def _build_tensor(self, transform, sequences, key, dim):
        """Convert input sequences to tensors."""
        tensors = []
        for sequence in sequences:
            tensors.append(transform(sequence[key]))

        return torch.stack(tensors, dim=dim).to(self._device)

    # ################## #
    # GAN Training steps #
    # ################## #

    def _transform(self, data):
        for properties in self._data_map.values():
            column_type = properties['type']
            if column_type in ('continuous', 'count'):
                value_idx, missing_idx = properties['indices']
                data[:, :, value_idx] = torch.tanh(data[:, :, value_idx])
                data[:, :, missing_idx] = torch.sigmoid(data[:, :, missing_idx])
            elif column_type in ('categorical', 'ordinal'):
                indices = list(properties['indices'].values())
                data[:, :, indices] = torch.nn.functional.softmax(data[:, :, indices])

        return data

    def _truncate(self, generated):
        end_flag = (generated[:, :, self._data_size] > 0.5).float()
        end_flag = end_flag.round()
        generated[:, :, self._data_size] = end_flag

        for sequence_idx in range(generated.shape[1]):
            # Pad with zeroes after end_flag == 1
            sequence = generated[:, sequence_idx]
            end_flag = sequence[:, self._data_size]
            if (end_flag == 1.0).any():
                cut_idx = end_flag.detach().cpu().numpy().argmax()
                sequence[cut_idx + 1 :] = 0.0

    def _generate(self, context, sequence_length=None):
        generated = self._generator(
            context=context,
            sequence_length=sequence_length or self._max_sequence_length,
        )

        generated = self._transform(generated)
        if not self._fixed_length:
            self._truncate(generated)

        return generated

    def _discriminator_step(self, discriminator, discriminator_opt, data_context, context):
        real_scores = discriminator(data_context)

        fake = self._generate(context)
        fake_context = _expand_context(fake, context)
        fake_scores = discriminator(fake_context)

        discriminator_score = -torch.mean(real_scores - fake_scores)

        discriminator_opt.zero_grad()
        discriminator_score.backward()
        discriminator_opt.step()

        for parameter in discriminator.parameters():
            parameter.data.clamp_(-0.01, 0.01)

        return discriminator_score

    def _generator_step(self, discriminator, generator_opt, context):
        fake = self._generate(context)
        fake_context = _expand_context(fake, context)
        generator_score = -torch.mean(discriminator(fake_context))

        generator_opt.zero_grad()
        generator_score.backward()
        generator_opt.step()

        return generator_score

    def _build_fit_artifacts(self):
        self._generator = BasicGenerator(
            context_size=self._context_size,
            latent_size=self._latent_size,
            hidden_size=self._hidden_size,
            data_size=self._model_data_size,
            device=self._device,
        ).to(self._device)

        discriminator = BasicDiscriminator(
            context_size=self._context_size,
            data_size=self._model_data_size,
            hidden_size=self._hidden_size,
        ).to(self._device)

        generator_opt = torch.optim.Adam(self._generator.parameters(), lr=self._gen_lr)
        discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=self._dis_lr)

        return discriminator, generator_opt, discriminator_opt

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
        self._analyze_data(sequences, context_types, data_types)

        data = self._build_tensor(self._data_to_tensor, sequences, 'data', dim=1)
        context = self._build_tensor(self._context_to_tensor, sequences, 'context', dim=0)
        data_context = _expand_context(data, context)

        discriminator, generator_opt, discriminator_opt = self._build_fit_artifacts()

        iterator = range(self._epochs)
        if self._verbose:
            iterator = tqdm(iterator)

        for epoch in iterator:
            discriminator_score = self._discriminator_step(
                discriminator=discriminator,
                discriminator_opt=discriminator_opt,
                data_context=data_context,
                context=context,
            )
            generator_score = self._generator_step(
                discriminator=discriminator,
                generator_opt=generator_opt,
                context=context,
            )

            if self._verbose:
                d_loss = discriminator_score.item()
                g_loss = generator_score.item()
                iterator.set_description(f'Epoch {epoch + 1} | D Loss {d_loss} | G Loss {g_loss}')

    def sample_sequence(self, context, sequence_length=None):
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
        context = self._context_to_tensor(context).unsqueeze(0).to(self._device)

        with torch.no_grad():
            generated = self._generate(context, sequence_length)
            if sequence_length is None:
                end_flag = generated[:, 0, -1]
                if (end_flag == 1.0).any():
                    cut_index = end_flag.cpu().numpy().argmax()
                    generated = generated[:cut_index, :, :]

            return self._tensor_to_data(generated)
