"""dganger Model."""
# pylint: disable-all

import logging

import numpy as np
import torch
from tqdm import tqdm

from deepecho.models.base import DeepEcho
from deepecho.models.utils import (
    build_tensor, context_to_tensor, data_to_tensor, index_map, tensor_to_data)

LOGGER = logging.getLogger(__name__)


def _expand_context(data, context):
    """Expand context over data.

    Expand context dimensions over ``data`` and concatenate both variables.

    Args:
        data (torch.tensor):
            Batch of data samples
        context (torch.tensor):
            Contextual information for each sample.

    Returns:
        torch.tensor:
            Both inpunts expanded and concatenated.
    """
    return torch.cat([
        data,
        context.unsqueeze(0).expand(data.shape[0], context.shape[0], context.shape[1])
    ], dim=2)


def _flat_and_concat(sequences, context):
    """Flat ``sequences`` and concat the flatten vector with `` context``."""
    sequences = sequences.permute(1, 0, 2)
    sequences = sequences.flatten(start_dim=1)
    sequences = torch.cat((sequences, context), 1)
    return sequences


def _normalize_per_sample(data, data_map):
    """Normalize sample's variables.

    Normalize all the sample's variables. Return a tensor that contains the mid point and
    the interval length for each sample.

    Args:
        data(torch.tensor):
            All the normalized data.
        data_map(dict)
            Information related position of the variables in ``data``.

    Returns:
        torch.tensor:
            Contains the mid point and the interval length for each sample.
    """
    minmax_tensors = None
    for key in data_map.keys():
        if data_map[key]['type'] in ('count', 'continuous'):
            minmax_tensor = torch.zeros([data.shape[1], 2])
            index_variable = data_map[key]['indices'][0]
            for row in range(data.shape[1]):
                sequence = data[:, row, index_variable]
                min_value = sequence.min()
                max_value = sequence.max()

                values_range = max_value - min_value
                offset = sequence - min_value
                sequence_norm = 2.0 * offset / values_range - 1.0
                data[:, row, index_variable] = sequence_norm

                mid_point = (max_value - min_value) / 2
                interval_length = max_value + min_value

                minmax_tensor[row, 0] = mid_point
                minmax_tensor[row, 1] = interval_length

            if minmax_tensors is None:
                minmax_tensors = minmax_tensor
            else:
                minmax_tensors = torch.cat((minmax_tensors, minmax_tensor), dim=1)

    return minmax_tensors


def _denormalize_per_sample(generated, minmax_generated, data_map):
    """Denormalize generated sample's variables.

    Denormalize all the sample's variables. Return a tensor that contains the denormalized
    values.

    Args:
        generated(torch.tensor):
            All the generated data.
        minmax_generated(torch.tensor):
            Generated information, containing the mid point and the range length of the data.
        data_map(dict)
            Information related position of the variables in ``generated``.

    Returns:
        torch.tensor
            ``Generated`` variables denormalized.
    """
    for key in data_map.keys():
        if data_map[key]['type'] in ('count', 'continuous'):
            index_variable = data_map[key]['indices'][0]
            for row in range(generated.shape[1]):
                sequence = generated[:, row, index_variable]
                mid_point = minmax_generated[row, 0]
                values_range = minmax_generated[row, 1]

                min_value = mid_point - (values_range) / 2

                denormalized_sequence = (sequence + 1) * values_range / 2.0 + min_value
                generated[:, row, index_variable] = denormalized_sequence

            minmax_generated = minmax_generated[:, 2:]

    return generated


class MinMaxGenerator(torch.nn.Module):
    """Min max Generator for the DoopleGANger model.

    This generator consist on a 4 lineal layers with
    the following schema:

        - The Generator takes as input a ``context`` vector.
        - The ``context`` vector is padded with ``latent_size`` random noise.
        - The network takes as input a tensor with shape
          ``(context_size + latent_size)``.
        - Generates a tensor of size ``(minmax_size)``.

    Args:
        context_size (int):
            Size of the contextual arrays.
        minmax_size(int):
            Amount of minmax variables.
        latent_size (int):
            Size of the random input vector. If None, equal to ``minmax_size``.
        hidden_size (int):
            Size of the communication between the RNN and the Linear layer.
        device (torch.device):
            Device to which this Module is associated to.

    Returns:
        torch.tensor:
            Generated min-max parameters.
    """

    def __init__(self, context_size, minmax_size, latent_size, hidden_size, device):
        super().__init__()
        self.latent_size = latent_size
        self.multilayer = torch.nn.Sequential(
            torch.nn.Linear(context_size + latent_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, minmax_size))

        self.device = device

    def forward(self, context=None):
        """Forward computation.

        Args:
            context (tensor):
                Context values to use for each generated sequence.
        """
        latent = torch.randn(
            size=(context.size(0), self.latent_size),
            device=self.device
        )
        latent = torch.cat((latent, context), 1)

        return self.multilayer(latent)


class MinMaxDiscriminator(torch.nn.Module):
    """MinMax discriminator for the DoopleGANger model.

    This discriminator consist on a 6 Linear layers with
    the following schema:

        - The Discriminator takes as input a collection of sequences that include
           min-max values and the context columns.
        - The network takes as input a tensor with shape
          ``(input_size + context_size)``  that outputs a single value for each sequence.

    Args:
        context_size (int):
            Number of values in the contextual arrays.
        minmax_size (int):
            Number of values in min-max arrays.
        hidden_size (int):
            Size of the communication between Linear layers.

    Returns:
        torch.tensor:
            Score.
    """

    def __init__(self, context_size, minmax_size, hidden_size):
        super().__init__()

        self.multilayer = torch.nn.Sequential(
            torch.nn.Linear(context_size + minmax_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=hidden_size),

            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=hidden_size),

            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=hidden_size),

            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_features=hidden_size),

            torch.nn.Linear(hidden_size, 1))

    def forward(self, sequences):
        """Forward computation.

        Args:
            sequences (tensor):
                Sequences of values with their context.
        """
        return self.multilayer(sequences)


class TimeSeriesGenerator(torch.nn.Module):
    """TimeSeriesGenerator for the DoopleGANger model.

    This generator consist on a RNN layer followed by a Linear layer with
    the following schema:

        - The Generator takes as input a ``sequence_length``, ``context`` vector and ``min-max``
          vector.
        - The ``context`` and ``min-max`` vectors are expanded over the ``sequence_lenght`` and
           padded with ``latent_size`` random noise.
        - RNN takes as input a tensor with shape ``(context_size + latent_size + minmax_size)``
          , and the current state of the RNN.
        - The RNN output is passed to the Linear layer that generates a tensor of size
          ``(sequence_length, data_size)``

    Args:
        context_size (int):
            Size of the contextual arrays.
        data_size (int):
            Amount of generated variables.
        minmax_size(int):
            Amount of minmax variables.
        latent_size (int):
            Size of the random input vector. If None, equal to ``encoded_data``.
        hidden_size (int):
            Size of the communication between the RNN and the Linear layer.
        device (torch.device):
            Device to which this Module is associated to.
    """

    def __init__(self,
                 context_size,
                 data_size,
                 latent_size,
                 minmax_size,
                 hidden_size,
                 device):

        super().__init__()
        self.latent_size = latent_size
        self.rnn = torch.nn.RNN(context_size + latent_size + minmax_size, hidden_size)
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
            device=self.device
        )
        latent = _expand_context(latent, context)

        rnn_out, _ = self.rnn(latent)
        return self.linear(rnn_out)


class TimeSeriesDiscriminator(torch.nn.Module):
    """Discriminator for the DoopleGANger time series model.

    In the original code exist the option to add batch_normalization layers after activation.

    This discriminator consist on a 6 Linear layers with
    the following schema:

        - The Discriminator takes as input a collection of sequences that include
           data, min-max values and the context columns.
        - The network takes as input a tensor with shape
          ``(data_size + input_size and context_size)``  that outputs a single value.

    Args:
        context_size (int):
            Number of values in the contextual arrays.
        minmax_size (int):
            Number of values in min-max arrays.
        sequence_length(int):
            Length of the sequence.
        data_size (int):
            Number of variables in the input data sequences.
        hidden_size (int):
            Size of the communication between Linear layers.

    Returns:
        torch.tensor:
            Score.
    """

    def __init__(self, context_size, minmax_size, sequence_length, data_size, hidden_size):
        super().__init__()

        self.multilayer = torch.nn.Sequential(
            torch.nn.Linear((data_size * sequence_length) + context_size + minmax_size,
                            hidden_size),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_size, 1))

    def forward(self, sequences):
        """Forward computation.

        Args:
            sequences (tensor):
                Sequences of values with their context.
        """
        return self.multilayer(sequences)


class DGANger(DeepEcho):
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
    _minmax_size = None
    _data_map = None
    _data_size = None
    _model_data_size = None
    _minmax_generator = None
    _timeseries_generator = None

    def __init__(self,
                 epochs=128,
                 latent_size=5,
                 hidden_size=200,
                 mm_gen_lr=1e-3,
                 ts_gen_lr=1e-3,
                 mm_dis_lr=1e-3,
                 ts_dis_lr=1e-3,
                 cuda=True,
                 verbose=True):
        self._epochs = epochs
        self._mm_gen_lr = mm_gen_lr
        self._ts_gen_lr = ts_gen_lr
        self._mm_dis_lr = mm_dis_lr
        self._ts_dis_lr = ts_dis_lr
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
        if verbose:
            print(self, 'instance created')

    def __repr__(self):
        return (
            "{}(epochs={}, latent_size={}, hidden_size={}, "
            "mm_gen_lr={}, mm_dis_lr={}, ts_gen_lr={}, ts_dis_lr={}, cuda='{}', verbose={})"
        ).format(
            self.__class__.__name__,
            self._epochs,
            self._latent_size,
            self._hidden_size,
            self._mm_gen_lr,
            self._mm_dis_lr,
            self._ts_gen_lr,
            self._ts_dis_lr,
            self._device,
            self._verbose,
        )

    # ########################### #
    # Preprocessing and preparing #
    # ########################### #

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

        self._context_map, self._context_size = index_map(context, context_types)

        # Concatenate all the data sequences together
        data = []
        for column in range(len(data_types)):
            data.append(sum([sequence['data'][column] for sequence in sequences], []))

        self._data_map, self._data_size = index_map(data, data_types)

        self._model_data_size = self._data_size + int(not self._fixed_length)

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
                data[:, :, indices] = torch.nn.functional.softmax(data[:, :, indices], dim=0)

        return data

    def _truncate(self, generated):
        end_flag = (generated[:, :, self._data_size] > 0.5).float().round()
        generated[:, :, self._data_size] = end_flag

        for sequence_idx in range(generated.shape[1]):
            # Pad with zeroes after end_flag == 1
            sequence = generated[:, sequence_idx]
            end_flag = sequence[:, self._data_size]
            if (end_flag == 1.0).any():
                cut_idx = end_flag.detach().cpu().numpy().argmax()
                sequence[cut_idx + 1:] = 0.0

    def _real_data_flatten(self, data_context):
        """Flat the data loaded from the dataset."""
        data = data_context[:, :, :self._data_size]
        data = data.permute(1, 0, 2)
        data = data.flatten(start_dim=1)
        context = data_context[0, :, self._data_size:]
        return torch.cat((data, context), 1)

    def _generate_timeseries(self, generator, context, sequence_length=None):
        """Generate time series data and apply different prostprocessing steps.

        ``_generate_timeseries`` execute the follow steps:
            - Generate time series samples.
            - Transform the generated samples with ``_tranform``.
            - If ``sequence_lengt`` is not ``None``, apply ``_truncate``.

        Args:
            generator(DGANger.TimeSeriesGenerator):
                - Generator class to generate samples.
            context(torch.tensor):
                - Context information to generate samples.
            sequence_length(int):
                Length of the sequence.

        Returns:
            torch.vector
                Generated vector transformed and truncated.
        """
        generated = generator(
            context=context,
            sequence_length=sequence_length or self._max_sequence_length,
        )

        generated = self._transform(generated)
        if not self._fixed_length:
            self._truncate(generated)

        return generated

    def _discriminator_step(self,
                            generator,
                            discriminator,
                            discriminator_opt,
                            context,
                            data_context):
        """Improve the ``discriminator`` network.

        ``generator`` generate fake data by ``context`` and ``minmax``,
        _discriminator_step execute the follow steps:
            - ``discriminator`` evaluates real data stored in the variable ``data_samples``.
            - Merge fake data and context.
            - Evaluate generated samples with ``discriminator``.
            - Calculate the loss.
            - Optimize ``generator`` with ``discriminator_opt``.

        Args:
            generator(MinMaxGenerator or TimeSeriesGenerator):
                - Generator class to train.
            discriminator(MinMaxDiscriminator or TimeSeriesDiscriminator):
                - Discriminator class to evaluate ``generator``.
            discriminator_opt(torch optim):
                - Method defined for optimization step.
            context(torch.tensor):
                - Context information related to generated sample.
                  Includes as much samples as is defined in ``self._batch_size``.
            min-max(torch.tensor):
            data_context(torch.tensor):
                - Data from the original dataset. Used to evaluate discriminator.
                - Min-max information related to generated sample.
                  Includes as much samples as is defined in ``self._batch_size``.

        Returns:
            torch.tensor()
                Discriminator score.
        """
        if isinstance(generator, MinMaxGenerator):
            real_scores = discriminator(data_context)
            fake = generator(context)
            fake_context = torch.cat((context, fake), 1)
        else:
            data_context = self._real_data_flatten(data_context)
            real_scores = discriminator(data_context)
            fake = self._generate_timeseries(generator, context)
            fake_context = _flat_and_concat(fake, context)
        fake_scores = discriminator(fake_context)

        discriminator_score = -torch.mean(real_scores - fake_scores)

        discriminator_opt.zero_grad()
        discriminator_score.backward()
        discriminator_opt.step()

        return discriminator_score

    def _generator_step(self, generator, discriminator, generator_opt, context):
        """Improve the ``generator`` network.

        _generator_step execute the follow steps:
            - ``generator`` generate fake data by ``context`` and ``minmax``.
            - Merge fake data and context.
            - Evaluate generated samples with ``discriminator``.
            - Optimize ``generator`` with ``generator_opt``.

        Args:
            generator(MinMaxGenerator or TimeSeriesGenerator):
                - Generator class to train.
            discriminator(MinMaxDiscriminator or TimeSeriesDiscriminator):
                - Discriminator class to evaluate ``generator``.
            generator_opt(torch optim):
                - Method defined for optimization step.
            context(torch.tensor):
                - Context information related to generated sample. Includes min-max.

        Returns:
            torch.tensor()
                Generator score.
        """
        if isinstance(generator, MinMaxGenerator):
            fake = generator(context)
            fake_context = torch.cat((context, fake), 1)
        else:
            fake = self._generate_timeseries(generator, context)
            fake_context = _flat_and_concat(fake, context)

        generator_score = -torch.mean(discriminator(fake_context))

        generator_opt.zero_grad()
        generator_score.backward()
        generator_opt.step()

        return generator_score

    def _build_minmax_gan_artifacts(self):
        """Inicialize discriminators networks, generators networks and optimizers.

        ``_build_minmax_gan_artifacts`` function is expected to:
            - Inicialize MinMaxGenerator as attribute of DGANger instance.
            - Inicialize MinMaxDiscriminator to return as a parameter.
            - Inicialize minmax_generator_opt to return as a parameter.
            - Inicialize minax_discriminator_opt to return as a parameter.

        Returns:
            Tuple:
                *``MinMaxDiscriminator``:MinMaxDiscriminator network.
                *``torch optim``: optimizer for MinMaxGenerator.
                *``torch optim``: optimizer for MinMaxDiscriminator.
        """
        self._minmax_generator = MinMaxGenerator(
            context_size=self._context_size,
            latent_size=self._latent_size,
            hidden_size=self._hidden_size,
            minmax_size=self._minmax_size,
            device=self._device,
        ).to(self._device)

        minmax_discriminator = MinMaxDiscriminator(
            context_size=self._context_size,
            minmax_size=self._minmax_size,
            hidden_size=self._hidden_size,
        ).to(self._device)

        minmax_generator_opt = torch.optim.Adam(
            self._minmax_generator.parameters(),
            lr=self._mm_gen_lr)

        minmax_discriminator_opt = torch.optim.Adam(
            minmax_discriminator.parameters(),
            lr=self._mm_dis_lr)

        return (
            minmax_discriminator,
            minmax_generator_opt,
            minmax_discriminator_opt,
        )

    def _build_timeseries_gan_artifacts(self):
        """Inicialize discriminators networks, generators networks and optimizers.

        ```_build_timeseries_gan_artifacts``` function is expected to:
            - Inicialize TimeSeriesGenerator as attribute of DGANger instance.
            - Inicialize TimeSeriesDiscriminator to return as a parameter.
            - Inicialize timeseries_generator_opt to return as a parameter.
            - Inicialize timeseries_discriminator_opt to return as a parameter.

        Returns:
            Tuple:
                *``TimeSeriesDiscriminator``:TimeSeriesDiscriminator network.
                *``torch optim``: optimizer for TimeSeriesGenerator.
                *``torch optim``: optimizer for TimeSeriesDiscriminator.
        """
        self._timeseries_generator = TimeSeriesGenerator(
            context_size=self._context_size,
            latent_size=self._latent_size,
            minmax_size=self._minmax_size,
            hidden_size=self._hidden_size,
            data_size=self._model_data_size,
            device=self._device,
        ).to(self._device)

        timeseries_discriminator = TimeSeriesDiscriminator(
            context_size=self._context_size,
            sequence_length=self._max_sequence_length,
            minmax_size=self._minmax_size,
            data_size=self._model_data_size,
            hidden_size=self._hidden_size,
        ).to(self._device)

        timeseries_generator_opt = torch.optim.Adam(
            self._timeseries_generator.parameters(),
            lr=self._ts_gen_lr)

        timeseries_discriminator_opt = torch.optim.Adam(
            timeseries_discriminator.parameters(),
            lr=self._ts_dis_lr)

        return (
            timeseries_discriminator,
            timeseries_generator_opt,
            timeseries_discriminator_opt
        )

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
        self._analyze_data(sequences, context_types, data_types)
        self._minmax_size = (data_types.count('continuous') + data_types.count('count')) * 2

        data = build_tensor(
            transform=data_to_tensor,
            sequences=sequences,
            key='data',
            dim=1,
            model_data_size=self._model_data_size,
            data_map=self._data_map,
            fixed_length=self._fixed_length,
            max_sequence_length=self._max_sequence_length
        ).to(self._device)

        context = build_tensor(
            transform=context_to_tensor,
            sequences=sequences,
            key='context',
            dim=0,
            context_size=self._context_size,
            context_map=self._context_map
        ).to(self._device)

        minmax_tensor = _normalize_per_sample(data, self._data_map)

        context_minmax = torch.cat((context, minmax_tensor), 1)
        data_context_minmax = _expand_context(data, context_minmax)

        (
            minmax_discriminator,
            minmax_generator_opt,
            minmax_discriminator_opt,
        ) = self._build_minmax_gan_artifacts()

        (
            timeseries_discriminator,
            timeseries_generator_opt,
            timeseries_discriminator_opt,
        ) = self._build_timeseries_gan_artifacts()

        minmax_iterator = range(self._epochs)
        if self._verbose:
            minmax_iterator = tqdm(minmax_iterator)

        for minmax_epoch in minmax_iterator:

            # Train min-max.
            minmax_discriminator_score = self._discriminator_step(
                generator=self._minmax_generator,
                discriminator=minmax_discriminator,
                discriminator_opt=minmax_discriminator_opt,
                data_context=context_minmax,
                context=context,
            )
            minmax_generator_score = self._generator_step(
                generator=self._minmax_generator,
                discriminator=minmax_discriminator,
                generator_opt=minmax_generator_opt,
                context=context,
            )

            if self._verbose:
                minmax_iterator.set_description(
                    'Epoch {} | D MM Loss {} | G MM Loss {}'.format(
                        minmax_epoch + 1,
                        minmax_discriminator_score.item(),
                        minmax_generator_score.item(),
                    )
                )

        timeseries_iterator = range(self._epochs)
        if self._verbose:
            timeseries_iterator = tqdm(timeseries_iterator)

        for timeseries_epoch in timeseries_iterator:

            # Train time series.
            timeseries_discriminator_score = self._discriminator_step(
                generator=self._timeseries_generator,
                discriminator=timeseries_discriminator,
                discriminator_opt=timeseries_discriminator_opt,
                data_context=data_context_minmax,
                context=context_minmax,
            )
            timeseries_generator_score = self._generator_step(
                generator=self._timeseries_generator,
                discriminator=timeseries_discriminator,
                generator_opt=timeseries_generator_opt,
                context=context_minmax,
            )

            if self._verbose:
                timeseries_iterator.set_description(
                    'Epoch {} | D TS Loss {} | G TS Loss {}'.format(
                        timeseries_epoch + 1,
                        timeseries_discriminator_score.item(),
                        timeseries_generator_score.item()
                    )
                )

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
        context_tensor = context_to_tensor(context, self._context_size, self._context_map)
        context = context_tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            minmax_generated = self._minmax_generator(context)
            context = torch.cat((context, minmax_generated), 1)
            generated = self._generate_timeseries(self._timeseries_generator, context)
            generated = _denormalize_per_sample(generated, minmax_generated, self._data_map)
            if sequence_length is None:
                end_flag = generated[:, 0, -1]
                if (end_flag == 1.0).any():
                    cut_index = end_flag.cpu().numpy().argmax()
                    generated = generated[:cut_index, :, :]

            return tensor_to_data(generated, self._data_map)
