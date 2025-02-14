"""Probabilistic autoregressive model."""

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from deepecho.models.base import DeepEcho

LOGGER = logging.getLogger(__name__)


class PARNet(torch.nn.Module):
    """PARModel ANN model."""

    def __init__(self, data_size, context_size, hidden_size=32):
        super(PARNet, self).__init__()
        self.context_size = context_size
        self.down = torch.nn.Linear(data_size + context_size, hidden_size)
        self.rnn = torch.nn.GRU(hidden_size, hidden_size)
        self.up = torch.nn.Linear(hidden_size, data_size)

    def forward(self, x, c):
        """Forward passing computation."""
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            if self.context_size:
                x = torch.cat(
                    [
                        x,
                        c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1]),
                    ],
                    dim=2,
                )

            x = self.down(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
            x, _ = self.rnn(x)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            x = self.up(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

        else:
            if self.context_size:
                x = torch.cat(
                    [
                        x,
                        c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1]),
                    ],
                    dim=2,
                )

            x = self.down(x)
            x, _ = self.rnn(x)
            x = self.up(x)

        return x


class PARModel(DeepEcho):
    """Probabilistic autoregressive model.

    1. Analyze data types.
        - Categorical/Ordinal: Create mapping.
        - Continuous/Timestamp: Normalize, etc.
        - Count: Compute min and range
    2. Data -> Tensor
        - Categorical/Ordinal: Create one-hot
        - Continuous/Timestamp: Copy into `mu` after normalize, set `var=0.0`
        - Count: Subtract min, divide by range, copy into `r`, set `p=0.0`
    3. Loss
        - Categorical/Ordinal: Cross entropy
        - Continuous/Timestamp: Gaussian likelihood
        - Count: Negative binomial (multiply param + value by range), evaluate loss.
    4. Sample (Tensor -> Tensor)
        - Categorical/Ordinal: Categorical distribution, store as one hot
        - Continuous/Timestamp: Gaussian sample, store into `mu`
        - Count: Negative binomial sample (multiply `r` by range?), divide by range
    5. Tensor -> Data
        - Categorical/Ordinal: Find the maximum value
        - Continuous/Timestamp: Rescale the value in `mu`
        - Count: Multiply by range, add min.

    Args:
        epochs (int):
            The number of epochs to train for. Defaults to 128.
        sample_size (int):
            The number of times to sample (before choosing and
            returning the sample which maximizes the likelihood).
            Defaults to 1.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        verbose (bool):
            Whether to print progress to console or not.
    """

    def __init__(self, epochs=128, sample_size=1, cuda=True, verbose=True):
        self.epochs = epochs
        self.sample_size = sample_size

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self.device = torch.device(device)
        self.verbose = verbose
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Loss'])

        LOGGER.info('%s instance created', self)

    def __repr__(self):
        """Return a representation of the class object."""
        return (
            f'{self.__class__.__name__}(epochs={self.epochs}, sample_size={self.sample_size},'
            f"cuda='{self.device}', verbose={self.verbose})"
        )

    def _idx_map(self, x, t):
        idx = 0
        idx_map = {}
        for i, t in enumerate(t):
            if t == 'continuous' or t == 'datetime':
                idx_map[i] = {
                    'type': t,
                    'mu': np.nanmean(x[i]),
                    'std': np.nanstd(x[i]),
                    'nulls': pd.isna(x[i]).any(),
                    'indices': (idx, idx + 1, idx + 2),
                }
                idx += 3

            elif t == 'count':
                idx_map[i] = {
                    'type': t,
                    'min': np.nanmin(x[i]),
                    'range': np.nanmax(x[i]) - np.nanmin(x[i]),
                    'nulls': pd.isna(x[i]).any(),
                    'indices': (idx, idx + 1, idx + 2),
                }
                idx += 3

            elif t == 'categorical' or t == 'ordinal':
                idx_map[i] = {'type': t, 'indices': {}}
                idx += 1
                for v in set(x[i]):
                    if pd.isna(v):
                        v = None

                    idx_map[i]['indices'][v] = idx
                    idx += 1

            else:
                raise ValueError(f'Unsupported type: {t}')

        return idx_map, idx

    def _build(self, sequences, context_types, data_types):
        contexts = [[] for _ in range(len(context_types))]
        data = [[] for _ in range(len(data_types))]
        min_length = np.inf
        max_length = -np.inf
        for sequence in sequences:
            sequence_data = sequence['data']
            sequence_context = sequence['context']
            sequence_length = len(sequence_data[0])
            min_length = min(min_length, sequence_length)
            max_length = max(max_length, sequence_length)

            for i in range(len(context_types)):
                contexts[i].append(sequence_context[i])
            for i in range(len(data_types)):
                data[i].extend(sequence_data[i])

        self._fixed_length = min_length == max_length
        self._min_length = min_length
        self._max_length = max_length

        self._ctx_map, self._ctx_dims = self._idx_map(contexts, context_types)
        self._data_map, self._data_dims = self._idx_map(data, data_types)
        self._data_map['<TOKEN>'] = {
            'type': 'categorical',
            'indices': {
                '<START>': self._data_dims,
                '<END>': self._data_dims + 1,
                '<BODY>': self._data_dims + 2,
            },
        }
        self._data_dims += 3

    def _data_to_tensor(self, data):
        seq_len = len(data[0])
        X = []

        x = torch.zeros(self._data_dims)
        x[self._data_map['<TOKEN>']['indices']['<START>']] = 1.0
        X.append(x)

        for i in range(seq_len):
            x = torch.zeros(self._data_dims)
            for key, props in self._data_map.items():
                if key == '<TOKEN>':
                    x[self._data_map['<TOKEN>']['indices']['<BODY>']] = 1.0

                elif props['type'] in ['continuous', 'timestamp']:
                    mu_idx, sigma_idx, missing_idx = props['indices']
                    if pd.isna(data[key][i]) or props['std'] == 0:
                        x[mu_idx] = 0.0
                    else:
                        x[mu_idx] = (data[key][i] - props['mu']) / props['std']

                    x[sigma_idx] = 0.0
                    x[missing_idx] = 1.0 if pd.isna(data[key][i]) else 0.0

                elif props['type'] in ['count']:
                    r_idx, p_idx, missing_idx = props['indices']
                    if pd.isna(data[key][i]) or props['range'] == 0:
                        x[r_idx] = 0.0
                    else:
                        x[r_idx] = (data[key][i] - props['min']) / props['range']

                    x[p_idx] = 0.0
                    x[missing_idx] = 1.0 if pd.isna(data[key][i]) else 0.0

                elif props['type'] in [
                    'categorical',
                    'ordinal',
                ]:  # categorical
                    value = data[key][i]
                    if pd.isna(value):
                        value = None
                    x[props['indices'][value]] = 1.0

                else:
                    raise ValueError()

            X.append(x)

        x = torch.zeros(self._data_dims)
        x[self._data_map['<TOKEN>']['indices']['<END>']] = 1.0
        X.append(x)

        return torch.stack(X, dim=0).to(self.device)

    def _context_to_tensor(self, context):
        if not self._ctx_dims:
            return None

        x = torch.zeros(self._ctx_dims)
        for key, props in self._ctx_map.items():
            if props['type'] in ['continuous', 'datetime']:
                mu_idx, sigma_idx, missing_idx = props['indices']
                x[mu_idx] = (
                    0.0
                    if (pd.isna(context[key]) or props['std'] == 0)
                    else (context[key] - props['mu']) / props['std']
                )
                x[sigma_idx] = 0.0
                x[missing_idx] = 1.0 if pd.isna(context[key]) else 0.0

            elif props['type'] in ['count']:
                r_idx, p_idx, missing_idx = props['indices']
                x[r_idx] = (
                    0.0
                    if (pd.isna(context[key]) or props['range'] == 0)
                    else (context[key] - props['min']) / props['range']
                )
                x[p_idx] = 0.0
                x[missing_idx] = 1.0 if pd.isna(context[key]) else 0.0

            elif props['type'] in ['categorical', 'ordinal']:
                value = context[key]
                if pd.isna(value):
                    value = None
                x[props['indices'][value]] = 1.0

            else:
                raise ValueError()

        return x.to(self.device)

    def fit_sequences(self, sequences, context_types, data_types):
        """Fit a model to the specified sequences.

        Args:
            sequences (list):
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
            context_types (list):
                List of strings indicating the type of each value in context.
                he value at `context[i]` must match the type specified by
                `context_types[i]`. Valid types include the following: `categorical`,
                `continuous`, `ordinal`, `count`, and `datetime`.
            data_types (list):
                List of strings indicating the type of each channel in data.
                Each value in the list at data[i] must match the type specified by
                `data_types[i]`. The valid types are the same as for `context_types`.
        """
        X, C = [], []
        self._build(sequences, context_types, data_types)
        for sequence in sequences:
            X.append(self._data_to_tensor(sequence['data']))
            C.append(self._context_to_tensor(sequence['context']))

        X = torch.nn.utils.rnn.pack_sequence(X, enforce_sorted=False).to(self.device)
        if self._ctx_dims:
            C = torch.stack(C, dim=0).to(self.device)

        self._model = PARNet(self._data_dims, self._ctx_dims).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            pbar_description = 'Loss ({loss:.3f})'
            iterator.set_description(pbar_description.format(loss=0))

        # Reset loss_values dataframe
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Loss'])

        X_padded, seq_len = torch.nn.utils.rnn.pad_packed_sequence(X)
        for epoch in iterator:
            Y = self._model(X, C)
            Y_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(Y)

            optimizer.zero_grad()
            loss = self._compute_loss(X_padded[1:, :, :], Y_padded[:-1, :, :], seq_len)
            loss.backward()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [epoch],
                'Loss': [loss.item()],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([
                    self.loss_values,
                    epoch_loss_df,
                ]).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(pbar_description.format(loss=loss.item()))

            optimizer.step()

    def _compute_loss(self, X_padded, Y_padded, seq_len):
        """Compute the loss between X and Y.

        Given X[i,:,:], the neural network predicts the value at the next
        timestep (i+1); this prediction is provided in Y[i,:,:]. This function
        returns the loss between the predicted and actual sequence.

        .. note::
            The `i`th time series (with padding removed) can be indexed with
            `X[:seq_len[i], i, :]`.

        Args:
            X_padded (tensor):
                This contains the input to the model.
            Y_padded (tensor):
                This contains the output of the model.
            seq_len (list):
                This list contains the length of each sequence.
        """
        log_likelihood = 0.0
        _, batch_size, _input_size = X_padded.shape

        for key, props in self._data_map.items():
            if props['type'] in ['continuous', 'timestamp']:
                mu_idx, sigma_idx, missing_idx = props['indices']
                mu = Y_padded[:, :, mu_idx]
                sigma = torch.nn.functional.softplus(Y_padded[:, :, sigma_idx])
                missing = torch.nn.LogSigmoid()(Y_padded[:, :, missing_idx])

                for i in range(batch_size):
                    dist = torch.distributions.normal.Normal(
                        mu[: seq_len[i], i], sigma[: seq_len[i], i]
                    )
                    log_likelihood += torch.sum(dist.log_prob(X_padded[-seq_len[i] :, i, mu_idx]))

                    p_true = X_padded[: seq_len[i], i, missing_idx]
                    p_pred = missing[: seq_len[i], i]
                    log_likelihood += torch.sum(p_true * p_pred)
                    log_likelihood += torch.sum((1.0 - p_true) * torch.log(1.0 - torch.exp(p_pred)))

            elif props['type'] in ['count']:
                r_idx, p_idx, missing_idx = props['indices']
                r = torch.nn.functional.softplus(Y_padded[:, :, r_idx]) * props['range']
                p = torch.sigmoid(Y_padded[:, :, p_idx])
                x = X_padded[:, :, r_idx] * props['range']
                missing = torch.nn.LogSigmoid()(Y_padded[:, :, missing_idx])

                for i in range(batch_size):
                    dist = torch.distributions.negative_binomial.NegativeBinomial(
                        r[: seq_len[i], i],
                        p[: seq_len[i], i],
                        validate_args=False,
                    )
                    log_likelihood += torch.sum(dist.log_prob(x[: seq_len[i], i]))

                    p_true = X_padded[: seq_len[i], i, missing_idx]
                    p_pred = missing[: seq_len[i], i]
                    log_likelihood += torch.sum(p_true * p_pred)
                    log_likelihood += torch.sum((1.0 - p_true) * torch.log(1.0 - torch.exp(p_pred)))

            elif props['type'] in ['categorical', 'ordinal']:
                idx = list(props['indices'].values())
                log_softmax = torch.nn.functional.log_softmax(Y_padded[:, :, idx], dim=2)

                for i in range(batch_size):
                    target = X_padded[: seq_len[i], i, idx]
                    predicted = log_softmax[: seq_len[i], i]
                    target = torch.argmax(target, dim=1).unsqueeze(dim=1)
                    log_likelihood += torch.sum(predicted.gather(dim=1, index=target))

            else:
                raise ValueError()

        return -log_likelihood / (batch_size * len(self._data_map) * batch_size)

    def _tensor_to_data(self, x):
        # Force CPU on x
        x = x.to(torch.device('cpu'))

        seq_len, batch_size, _ = x.shape
        assert batch_size == 1

        data = [None] * (len(self._data_map) - 1)
        for key, props in self._data_map.items():
            if key == '<TOKEN>':
                continue

            data[key] = []
            for i in range(seq_len):
                if props['type'] in ['continuous', 'datetime']:
                    mu_idx, _sigma_idx, missing_idx = props['indices']
                    if (x[i, 0, missing_idx] > 0) and props['nulls']:
                        data[key].append(None)
                    else:
                        data[key].append(x[i, 0, mu_idx].item() * props['std'] + props['mu'])

                elif props['type'] in ['count']:
                    r_idx, _p_idx, missing_idx = props['indices']
                    if x[i, 0, missing_idx] > 0 and props['nulls']:
                        data[key].append(None)
                    else:
                        sample = x[i, 0, r_idx].item() * props['range'] + props['min']
                        data[key].append(int(sample))

                elif props['type'] in ['categorical', 'ordinal']:
                    ml_value, max_x = None, float('-inf')
                    for value, idx in props['indices'].items():
                        if x[i, 0, idx] > max_x:
                            max_x = x[i, 0, idx]
                            ml_value = value

                    data[key].append(ml_value)

                else:
                    raise ValueError()

        return data

    def _sample_state(self, x):
        log_likelihood = 0.0
        seq_len, batch_size, _input_size = x.shape
        assert seq_len == 1 and batch_size == 1

        for key, props in self._data_map.items():
            if props['type'] in ['continuous', 'timestamp']:
                mu_idx, sigma_idx, missing_idx = props['indices']
                mu = x[0, 0, mu_idx]
                sigma = torch.nn.functional.softplus(x[0, 0, sigma_idx])
                dist = torch.distributions.normal.Normal(mu, sigma)
                x[0, 0, mu_idx] = dist.sample()
                x[0, 0, sigma_idx] = 0.0
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, mu_idx]))

                dist = torch.distributions.Bernoulli(torch.sigmoid(x[0, 0, missing_idx]))
                x[0, 0, missing_idx] = dist.sample()
                x[0, 0, mu_idx] = x[0, 0, mu_idx] * (1.0 - x[0, 0, missing_idx])
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, missing_idx]))

            elif props['type'] in ['count']:
                r_idx, p_idx, missing_idx = props['indices']
                r = torch.nn.functional.softplus(x[0, 0, r_idx]) * props['range']
                p = torch.sigmoid(x[0, 0, p_idx])
                dist = torch.distributions.negative_binomial.NegativeBinomial(r, p)
                x[0, 0, r_idx] = dist.sample()
                x[0, 0, p_idx] = 0.0
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, r_idx]))
                x[0, 0, r_idx] /= props['range']

                dist = torch.distributions.Bernoulli(torch.sigmoid(x[0, 0, missing_idx]))
                x[0, 0, missing_idx] = dist.sample()
                x[0, 0, r_idx] = x[0, 0, r_idx] * (1.0 - x[0, 0, missing_idx])
                log_likelihood += torch.sum(dist.log_prob(x[0, 0, missing_idx]))

            elif props['type'] in ['categorical', 'ordinal']:
                idx = list(props['indices'].values())
                p = torch.nn.functional.softmax(x[0, 0, idx], dim=0)
                x_new = torch.zeros(p.size()).to(self.device)
                x_new.scatter_(dim=0, index=torch.multinomial(p, 1), value=1)
                x[0, 0, idx] = x_new
                log_likelihood += torch.sum(torch.log(p) * x_new)

            else:
                raise ValueError()

        return x, log_likelihood

    def _sample_sequence(self, context, min_length, max_length):
        log_likelihood = 0.0

        x = torch.zeros(self._data_dims).to(self.device)
        x[self._data_map['<TOKEN>']['indices']['<START>']] = 1.0
        x = x.unsqueeze(0).unsqueeze(0)

        for step in range(max_length):
            next_x, ll = self._sample_state(self._model(x, context)[-1:, :, :])
            x = torch.cat([x, next_x], dim=0)
            log_likelihood += ll
            if next_x[0, 0, self._data_map['<TOKEN>']['indices']['<END>']] > 0.0:
                if min_length <= step + 1 <= max_length:
                    break  # received end token

                next_x[0, 0, self._data_map['<TOKEN>']['indices']['<BODY>']] = 1.0
                next_x[0, 0, self._data_map['<TOKEN>']['indices']['<END>']] = 0.0

        return x[1:, :, :], log_likelihood

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
        if sequence_length is not None:
            min_length = max_length = sequence_length
        else:
            min_length = self._min_length
            max_length = self._max_length

        if self._ctx_dims:
            context = self._context_to_tensor(context).unsqueeze(0)
        else:
            context = None

        best_x, best_ll = None, float('-inf')
        for _ in range(self.sample_size):
            with torch.no_grad():
                x, log_likelihood = self._sample_sequence(context, min_length, max_length)

            if log_likelihood > best_ll:
                best_x = x
                best_ll = log_likelihood

        return self._tensor_to_data(best_x)
