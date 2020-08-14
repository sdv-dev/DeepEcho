import numpy as np
import torch
from tqdm import tqdm

from deepecho.base import DeepEcho


class Generator(torch.nn.Module):
    """
    0. Generator(input_size, latent_size, hidden_size, output_size)
    1. forward(context_vector)
        - internally generates noise?
        - returns *unnormalized* values
    """

    def __init__(self, context_size, latent_size, hidden_size, data_size):
        """
        Args:
            context_size: ...
            latent_size: ...
            hidden_size: ...
            output_size: ...
        """
        super(Generator, self).__init__()
        self.context_size = context_size
        self.latent_size = latent_size
        self.rnn = torch.nn.GRU(context_size + latent_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, data_size)

    def forward(self, sequence_length=None, context=None, batch_size=None):
        """
        # duplicate context over time: (seq_len, batch, context_size)
        # generate gaussian noise: (seq_len, batch, latent_size)
        # rnn takes as input: (seq_len, batch, context_size+latent_size)
        # rnn puts as output: (seq_len, batch, hidden_size)
        # linear scales to  : (seq_len, batch, data_size)

        # if fixed_length, then stop sequence generation after N tokens
        # if not specified, then generate to max length and truncate?
        """
        if self.context_size:
            x = torch.randn(size=(sequence_length, context.size(0), self.latent_size))
            x = torch.cat([
                x,
                context.unsqueeze(0).expand(sequence_length, context.shape[0], context.shape[1])
            ], dim=2)
        else:
            x = torch.randn(size=(sequence_length, batch_size, self.latent_size))

        x, _ = self.rnn(x)
        x = self.linear(x[:, :, :])
        return x


class Discriminator(torch.nn.Module):

    def __init__(self, context_size, data_size, hidden_size):
        """
        Args:
            context_size: ...
            data_size: ...
            hidden_size: ...
        """
        super(Discriminator, self).__init__()
        self.context_size = context_size
        self.rnn = torch.nn.GRU(context_size + data_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, c, x):
        """
        # duplicate context over time: (seq_len, batch, context_size)
        # rnn takes as input: (seq_len, batch, context_size+data_size)
        # rnn puts as output: (seq_len, batch, hidden_size)
        # final state -> linear  : (batch, 1)
        """
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            if self.context_size:
                x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
                x = torch.cat([
                    x,
                    c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1])
                ], dim=2)
                x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

            x, _ = self.rnn(x)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            x = self.linear(x[-1, :, :])

        else:
            if self.context_size:
                x = torch.cat([
                    x,
                    c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1])
                ], dim=2)

            x, _ = self.rnn(x)
            x = self.linear(x[-1, :, :])

        return x


class GANModel(DeepEcho):
    """Basic GAN model.

    0. Normalize continuous values to [-1.0, 1.0].
    1. Map categorical values to one-hot.
    1. Define a generator that takes context vector -> new sequence.
    2. Transforms
        - apply sigmoid to continuous/count/datetime
        - apply softmax to categorical/ordinal
    3. Define a discriminator that takes sequence + context -> score.
    """

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
        # transform context, data
        X, C = [], []
        self._build(sequences, context_types, data_types)
        for sequence in sequences:
            X.append(self._data_to_tensor(sequence['data']))
            C.append(self._context_to_tensor(sequence['context']))
        batch_size = len(X)
        X = torch.nn.utils.rnn.pack_sequence(X, enforce_sorted=False)
        if self._ctx_dims:
            C = torch.stack(C, dim=0)
        else:
            C = None

        G = Generator(
            context_size=self._ctx_dims,
            latent_size=32,
            hidden_size=16,
            data_size=self._data_dims)
        D = Discriminator(context_size=self._ctx_dims, data_size=self._data_dims, hidden_size=16)
        G_opt = torch.optim.Adam(G.parameters(), lr=1e-3)
        D_opt = torch.optim.Adam(D.parameters(), lr=1e-3)

        iterator = tqdm(range(1024))
        for epoch in iterator:
            syn_X = G(sequence_length=self._fixed_length, context=C, batch_size=batch_size)
            syn_X = self._transform(syn_X)
            D_score = torch.mean(D(C, X) - D(C, syn_X))

            D_opt.zero_grad()
            D_score.backward()
            D_opt.step()

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            syn_X = G(sequence_length=self._fixed_length, context=C, batch_size=batch_size)
            syn_X = self._transform(syn_X)
            G_score = torch.mean(D(C, syn_X))

            G_opt.zero_grad()
            G_score.backward()
            G_opt.step()

            iterator.set_description(
                "Epoch %s | D Loss %s | G Loss %s" %
                (epoch, D_score.item(), G_score.item()))

        self.G = G

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
        C = None
        if context:
            C = self._context_to_tensor(context).unsqueeze(0)
        X = self.G(sequence_length=self._fixed_length, context=C, batch_size=1)
        X = self._transform(X)
        return self._tensor_to_data(X)

    def _get_fixed_length(self, sequences):
        fixed_length = len(sequences[0]['data'][0])
        for sequence in sequences:
            if len(sequence['data'][0]) != fixed_length:
                return None

        return fixed_length

    def _build(self, sequences, context_types, data_types):
        self._fixed_length = self._get_fixed_length(sequences)

        contexts = []
        for i in range(len(context_types)):
            contexts.append([sequence['context'][i] for sequence in sequences])
        self._ctx_map, self._ctx_dims = self._idx_map(contexts, context_types)

        data = []
        for i in range(len(data_types)):
            data.append(sum([sequence['data'][i] for sequence in sequences], []))
        self._data_map, self._data_dims = self._idx_map(data, data_types)

    def _idx_map(self, x, t):
        idx = 0
        idx_map = {}
        for i, t in enumerate(t):
            if t == 'continuous' or t == 'datetime' or t == 'count':
                idx_map[i] = {
                    'type': t,
                    'min': np.min(x[i]),
                    'max': np.max(x[i]),
                    'indices': (idx, idx + 1)
                }
                idx += 2

            elif t == 'categorical' or t == 'ordinal':
                idx_map[i] = {
                    'type': t,
                    'indices': {}
                }
                for v in set(x[i]):
                    idx_map[i]['indices'][v] = idx
                    idx += 1

            else:
                raise ValueError('Unsupported type: {}'.format(t))

        return idx_map, idx

    def _data_to_tensor(self, data):
        seq_len = len(data[0])
        X = []

        for i in range(seq_len):
            x = torch.zeros(self._data_dims)
            for key, props in self._data_map.items():
                if props['type'] in ['continuous', 'timestamp', 'count']:
                    mu_idx, missing_idx = props['indices']
                    x[mu_idx] = 0.0 if data[key][i] is None else (
                        2.0 * (data[key][i] - props['min']) / (props['max'] - props['min']) - 1.0)
                    x[missing_idx] = 1.0 if data[key][i] is None else 0.0

                elif props['type'] in ['categorical', 'ordinal']:   # categorical
                    x[props['indices'][data[key][i]]] = 1.0

                else:
                    raise ValueError()

            X.append(x)

        X.append(x)
        return torch.stack(X, dim=0)

    def _context_to_tensor(self, context):
        if not self._ctx_dims:
            return None

        x = torch.zeros(self._ctx_dims)
        for key, props in self._ctx_map.items():
            if props['type'] in ['continuous', 'datetime', 'count']:
                mu_idx, missing_idx = props['indices']
                x[mu_idx] = 0.0 if context[key] is None else (
                    2.0 * (context[key] - props['min']) / (props['max'] - props['min']) - 1.0)
                x[missing_idx] = 1.0 if np.isnan(context[key]) else 0.0

            elif props['type'] in ['categorical', 'ordinal']:
                x[props['indices'][context[key]]] = 1.0

            else:
                raise ValueError()

        return x

    def _transform(self, x):
        for key, props in self._data_map.items():
            if props['type'] in ['continuous', 'timestamp', 'count']:
                mu_idx, missing_idx = props['indices']
                x[:, :, mu_idx] = torch.tanh(x[:, :, mu_idx])
                x[:, :, missing_idx] = torch.sigmoid(x[:, :, missing_idx])

            elif props['type'] in ['categorical', 'ordinal']:   # categorical
                idx = list(props['indices'].values())
                x[:, :, idx] = torch.nn.functional.softmax(x[:, :, idx])

        return x

    def _tensor_to_data(self, x):
        seq_len, batch_size, _ = x.shape
        assert batch_size == 1

        data = [None] * len(self._data_map)
        for key, props in self._data_map.items():
            if key == '<TOKEN>':
                continue

            data[key] = []
            for i in range(seq_len):
                if props['type'] in ['continuous', 'datetime', 'count']:
                    mu_idx, missing_idx = props['indices']
                    if x[i, 0, missing_idx] > 0.5:
                        data[key].append(None)
                    else:
                        data[key].append(((x[i, 0, mu_idx].item() + 1.0) / 2.0) *
                                         (props['max'] - props['min']) + props['min'])

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
