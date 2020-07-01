import numpy as np
import torch
from tqdm import tqdm

from deepecho.models.base import BaseModel


class PARModel(BaseModel):
    """Probabilistic autoregressive model.
    """

    def __init__(self, nb_epochs=64, max_seq_len=100, sample_size=1):
        """Initialize PARModel.

        Args:
            nb_epochs: The number of epochs to train for.
            max_seq_len: The maximum length of the sequence (if variable).
            sample_size: The number of times to sample (before choosing and
                returning the sample which maximizes the likelihood).
        """
        self.nb_epochs = nb_epochs
        self.max_seq_len = max_seq_len
        self.sample_size = sample_size

    def fit(self, sequences, context_types, data_types):
        X, C = [], []
        self._build(sequences, context_types, data_types)
        for sequence in sequences:
            X.append(self._data_to_tensor(sequence["data"]))
            C.append(self._context_to_tensor(sequence["context"]))
        X = torch.nn.utils.rnn.pack_sequence(X, enforce_sorted=False)
        if self._ctx_dims:
            C = torch.stack(C, dim=0)

        iterator = tqdm(range(self.nb_epochs))
        self._model = PARNet(self._data_dims, self._ctx_dims)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        for epoch in iterator:
            Y = self._model(X, C)
            X_padded, seq_len = torch.nn.utils.rnn.pad_packed_sequence(X)
            Y_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(Y)
            X_padded, Y_padded = X_padded[1:, :, :], Y_padded[:-1, :, :]

            optimizer.zero_grad()
            loss = self._compute_loss(X_padded, Y_padded, seq_len)
            loss.backward()
            iterator.set_description("Epoch %s | Loss %s" % (epoch, loss.item()))
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
            X_padded: This contains the input to the model.
            Y_padded: This contains the output of the model.
            seq_len: This list contains the length of each sequence.
        """
        log_likelihood = 0.0
        _, batch_size, input_size = X_padded.shape

        for key, indices in self._data_map.items():
            if isinstance(indices, tuple):  # continuous
                mu_idx, sigma_idx, missing_idx = indices
                for i in range(batch_size):
                    mu = Y_padded[:seq_len[i], i, mu_idx]
                    sigma = torch.nn.functional.softplus(Y_padded[:seq_len[i], i, sigma_idx])
                    x = X_padded[-seq_len[i]:, i, mu_idx]
                    dist = torch.distributions.normal.Normal(mu, sigma)
                    log_likelihood += torch.sum(dist.log_prob(x))

                    p_true = X_padded[:seq_len[i], i, missing_idx]
                    p_pred = torch.nn.LogSigmoid()(Y_padded[:seq_len[i], i, missing_idx])
                    log_likelihood += torch.sum(p_true * p_pred) + torch.sum(
                        (1.0 - p_true) * torch.log(1.0 - torch.exp(p_pred)))

            elif isinstance(indices, dict):   # categorical
                for i in range(batch_size):
                    idx = list(indices.values())
                    predicted, target = Y_padded[:seq_len[i],
                                                 i, idx], X_padded[:seq_len[i], i, idx]
                    predicted = torch.nn.functional.log_softmax(predicted, dim=1)
                    target = torch.argmax(target, dim=1).unsqueeze(dim=1)
                    log_likelihood += torch.sum(predicted.gather(dim=1, index=target))

            else:
                raise ValueError()
        return -log_likelihood / (batch_size * len(self._data_map) * batch_size)

    def sample(self, context):
        seq_len = self.max_seq_len
        if self._fixed_length:
            seq_len = self._fixed_length

        if self._ctx_dims:
            c = self._context_to_tensor(context).unsqueeze(0)
        else:
            c = None

        def sample_state(x):
            log_likelihood = 0.0
            seq_len, batch_size, input_size = x.shape
            assert seq_len == 1 and batch_size == 1

            for key, indices in self._data_map.items():
                if isinstance(indices, tuple):  # continuous
                    mu_idx, sigma_idx, missing_idx = indices
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

                elif isinstance(indices, dict):   # categorical
                    idx = list(indices.values())
                    p = torch.nn.functional.softmax(x[0, 0, idx], dim=0)
                    x_new = torch.zeros(p.size())
                    x_new.scatter_(dim=0, index=torch.multinomial(p, 1), value=1)
                    x[0, 0, idx] = x_new
                    log_likelihood += torch.sum(torch.log(p) * x_new)

                else:
                    raise ValueError()

            return x, log_likelihood

        def sample_sequence():
            log_likelihood = 0.0

            x = torch.zeros(self._data_dims)
            x[self._data_map["<TOKEN>"]["<START>"]] = 1.0
            x = x.unsqueeze(0).unsqueeze(0)

            for _ in range(seq_len):
                next_x, ll = sample_state(self._model(x, c)[-1:, :, :])
                x = torch.cat([x, next_x], dim=0)
                log_likelihood += ll
                if next_x[0, 0, self._data_map["<TOKEN>"]["<END>"]] > 0.0:
                    if not self._fixed_length:
                        break  # received end token
                    next_x[0, 0, self._data_map["<TOKEN>"]["<BODY>"]] = 1.0
                    next_x[0, 0, self._data_map["<TOKEN>"]["<END>"]] = 0.0
            
            return x[1:, :, :], log_likelihood

        best_x, best_ll = None, float("-inf")
        for _ in range(self.sample_size):
            x, log_likelihood = sample_sequence()
            if log_likelihood > best_ll:
                best_x = x
                best_ll = log_likelihood
        return self._tensor_to_data(best_x)

    def _build(self, sequences, context_types, data_types):
        contexts = []
        for i in range(len(context_types)):
            contexts.append([sequence["context"][i] for sequence in sequences])
        self._ctx_map, self._ctx_dims = self._idx_map(contexts, context_types)

        data = []
        for i in range(len(data_types)):
            data.append(sum([sequence["data"][i] for sequence in sequences], []))
        self._data_map, self._data_dims = self._idx_map(data, data_types)
        self._data_map["<TOKEN>"] = {
            "<START>": self._data_dims,
            "<END>": self._data_dims + 1,
            "<BODY>": self._data_dims + 2
        }
        self._data_dims += 3

        self._fixed_length = len(sequences[0]["data"][0])
        for sequence in sequences:
            if len(sequence["data"][0]) != self._fixed_length:
                self._fixed_length = None
                break

    def _idx_map(self, x, t):
        idx = 0
        idx_map = {}
        for i, t in enumerate(t):
            if t == "continuous" or t == "datetime" or t == "count":
                idx_map[i] = (idx, idx + 1, idx + 2)
                idx += 3
            elif t == "categorical" or t == "ordinal":
                idx_map[i] = {None: idx}
                idx += 1
                for v in set(x[i]):
                    idx_map[i][v] = idx
                    idx += 1
            else:
                raise ValueError("Unsupported type: %s" % t)
        return idx_map, idx

    def _data_to_tensor(self, data):
        seq_len = len(data[0])
        X = []

        x = torch.zeros(self._data_dims)
        x[self._data_map["<TOKEN>"]["<START>"]] = 1.0
        X.append(x)

        for i in range(seq_len):
            x = torch.zeros(self._data_dims)
            for key, indices in self._data_map.items():
                if key == "<TOKEN>":
                    x[self._data_map["<TOKEN>"]["<BODY>"]] = 1.0
                elif isinstance(indices, tuple):  # continuous
                    mu_idx, sigma_idx, missing_idx = indices
                    x[mu_idx] = 0.0 if data[key][i] is None else data[key][i]
                    x[sigma_idx] = 0.0
                    x[missing_idx] = 1.0 if data[key][i] is None else 0.0
                elif isinstance(indices, dict):   # categorical
                    x[indices[data[key][i]]] = 1.0
                else:
                    raise ValueError()
            X.append(x)

        x = torch.zeros(self._data_dims)
        x[self._data_map["<TOKEN>"]["<END>"]] = 1.0
        X.append(x)

        return torch.stack(X, dim=0)

    def _context_to_tensor(self, context):
        if not self._ctx_dims:
            return None
        x = torch.zeros(self._ctx_dims)
        for key, indices in self._ctx_map.items():
            if isinstance(indices, tuple):  # continuous
                mu_idx, sigma_idx, missing_idx = indices
                x[mu_idx] = 0.0 if np.isnan(context[key]) else context[key]
                x[sigma_idx] = 0.0
                x[missing_idx] = 1.0 if np.isnan(context[key]) else 0.0
            elif isinstance(indices, dict):  # categorical
                x[indices[context[key]]] = 1.0
            else:
                raise ValueError()
        return x

    def _tensor_to_data(self, x):
        seq_len, batch_size, _ = x.shape
        assert batch_size == 1

        data = [None] * (len(self._data_map) - 1)
        for key, indices in self._data_map.items():
            if key == "<TOKEN>":
                continue
            data[key] = []
            for i in range(seq_len):
                if isinstance(indices, tuple):  # continuous
                    mu_idx, sigma_idx, missing_idx = indices
                    if x[i, 0, missing_idx] > 0:
                        data[key].append(None)
                    else:
                        data[key].append(x[i, 0, mu_idx].item())

                elif isinstance(indices, dict):   # categorical
                    ml_value, max_x = None, float("-inf")
                    for value, idx in indices.items():
                        if x[i, 0, idx] > max_x:
                            max_x = x[i, 0, idx]
                            ml_value = value
                    data[key].append(ml_value)

                else:
                    raise ValueError()

        return data


class PARNet(torch.nn.Module):

    def __init__(self, data_size, context_size, hidden_size=32):
        super(PARNet, self).__init__()
        self.context_size = context_size
        self.down = torch.nn.Linear(data_size + context_size, hidden_size)
        self.rnn = torch.nn.GRU(hidden_size, hidden_size)
        self.up = torch.nn.Linear(hidden_size, data_size)

    def forward(self, x, c):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            if self.context_size:
                x = torch.cat([
                    x,
                    c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1])
                ], dim=2)
            x = self.down(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
            x, _ = self.rnn(x)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            x = self.up(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        else:
            if self.context_size:
                x = torch.cat([
                    x,
                    c.unsqueeze(0).expand(x.shape[0], c.shape[0], c.shape[1])
                ], dim=2)
            x = self.down(x)
            x, _ = self.rnn(x)
            x = self.up(x)
        return x
