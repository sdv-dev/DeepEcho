import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from deepecho import Dataset
from deepecho.model.base import Model


class Gamma(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Gamma, self).__init__()
        self.rnn = torch.nn.GRU(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            x = self.linear(x)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        else:
            x = self.linear(x)
        return x


class GammaModel(Model):
    """No time index, multiple entities, variable length.
    """

    @classmethod
    def check(cls, dataset):
        if dataset.time_idx:
            raise ValueError("This model doesn't support time indices.")
        if not dataset.entity_idx:
            raise ValueError("This model requires multiple entiies.")
        if dataset.fixed_length:
            raise ValueError("This model requires a variable length.")

    def fit(self, dataset):
        self.check(dataset)

        X = []
        self._build(dataset)
        for _, df in dataset.df.groupby(dataset.entity_idx):
            df = df.drop(dataset.entity_idx, axis=1)
            X.append(self._from_df(df))
        X = torch.nn.utils.rnn.pack_sequence(X, enforce_sorted=False)

        self._start = torch.randn(size=(self._input_size,), requires_grad=True)
        self._model = Gamma(self._input_size, 16, self._output_size)

        iterator = tqdm(range(1024))
        optimizer = torch.optim.Adam(list(self._model.parameters()) + [self._start], lr=1e-3)
        for epoch in iterator:
            Y = self._model(X)
            X_padded, seq_len = torch.nn.utils.rnn.pad_packed_sequence(X)
            Y_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(Y)

            log_likelihood = 0.0
            max_seq_len, batch_size, input_size = X_padded.shape

            for _, (mu_idx, logvar_idx) in self._continuous.items():
                for i in range(batch_size):
                    mu = Y_padded[-seq_len[i]:, i, mu_idx]
                    sigma = torch.nn.functional.softplus(Y_padded[-seq_len[i]:, i, logvar_idx])
                    x = X_padded[-seq_len[i]:, i, mu_idx]
                    dist = torch.distributions.normal.Normal(mu, sigma)
                    log_likelihood += torch.sum(dist.log_prob(x))

            for _, indices in self._categorical.items():
                for i in range(batch_size):
                    idx = list(indices.values())
                    predicted, target = Y_padded[-seq_len[i]:,
                                                 i, idx], X_padded[-seq_len[i]:, i, idx]
                    predicted = torch.nn.functional.log_softmax(predicted, dim=1)
                    target = torch.argmax(target, dim=1).unsqueeze(dim=1)
                    log_likelihood += torch.sum(predicted.gather(dim=1, index=target))

            log_likelihood /= batch_size

            optimizer.zero_grad()
            (-log_likelihood).backward()
            iterator.set_description(
                "Epoch %s | Log Likelihood %s" %
                (epoch, log_likelihood.item()))
            optimizer.step()

    def sample(self):
        dataframes = []

        def sample_sequence():
            log_likelihood = 0.0
            x = self._start.unsqueeze(0).unsqueeze(0)
            for _ in range(100):
                next_x, ll = self._from_latent(self._model(x)[-1:, :, :])
                x = torch.cat([x, next_x], dim=0)
                log_likelihood += ll
                if next_x[0, 0, self._categorical["<CONTEXT>"]["<END>"]] > 0.0:
                    break  # received end token
            return x[1:, :, :], log_likelihood

        for entity_id in range(self._nb_entities):
            best_x, best_ll = None, float("-inf")
            for _ in range(5):
                x, log_likelihood = sample_sequence()
                if log_likelihood > best_ll:
                    best_x = x
                    best_ll = log_likelihood
            df = self._from_input(best_x)
            df[self._entity_idx] = entity_id
            dataframes.append(df)

        return Dataset(
            pd.concat(dataframes, axis=0),
            entity_idx=self._entity_idx,
            fixed_length=self._seq_length
        )

    def _build(self, dataset):
        """Build internal data structures for transforms.
        """
        df = dataset.df.drop(dataset.entity_idx, axis=1)
        self._seq_length = dataset.fixed_length
        self._column_names = df.columns
        self._entity_idx = dataset.entity_idx
        self._nb_entities = dataset.df[dataset.entity_idx].nunique()

        idx = 0
        self._continuous = {}
        self._categorical = {}
        for column in df.columns:
            if df[column].dtype == np.float64:
                self._continuous[column] = (idx, idx + 1)
                idx += 2
            elif df[column].dtype == np.object:
                self._categorical[column] = {}
                for value in set(df[column]):
                    self._categorical[column][value] = idx
                    idx += 1
            else:
                raise ValueError("Unsupported type.")
        self._categorical["<CONTEXT>"] = {
            "<START>": idx,
            "<END>": idx + 1,
            "<BODY>": idx + 2
        }
        self._input_size = idx + 3
        self._output_size = self._input_size

    def _from_df(self, df):
        """Transform from dataframe to input space.

        This maps categorical columns from into a one-hot-like representation.
        """
        X = []

        x = torch.zeros(self._input_size)
        x[self._categorical["<CONTEXT>"]["<START>"]] = 1.0
        X.append(x)

        for _, row in df.iterrows():
            x = torch.zeros(self._input_size)
            for key, value in row.items():
                if key in self._continuous:
                    x[self._continuous[key][0]] = value
                elif key in self._categorical:
                    x[self._categorical[key][value]] = 1.0
                else:
                    raise ValueError("Unknown column %s" % key)
            x[self._categorical["<CONTEXT>"]["<BODY>"]] = 1.0
            X.append(x)

        x = torch.zeros(self._input_size)
        x[self._categorical["<CONTEXT>"]["<END>"]] = 1.0
        X.append(x)

        return torch.stack(X, dim=0)  # (L, C_in)

    def _from_latent(self, x):
        """Transform from latent space to input space.

        This samples from the distributions specified by the latent vector to
        generate a sample in the input space.
        """
        log_likelihood = 0.0
        seq_len, batch_size, input_size = x.shape

        for _, (mu_idx, logvar_idx) in self._continuous.items():
            for i in range(seq_len):
                dist = torch.distributions.normal.Normal(
                    x[i, :, mu_idx], torch.nn.functional.softplus(x[i, :, logvar_idx]))
                x[i, :, mu_idx] = dist.sample()
                x[i, :, logvar_idx] = 0.0
                log_likelihood += torch.sum(dist.log_prob(x[i, :, mu_idx]))

        for _, indices in self._categorical.items():
            idx = list(indices.values())
            for i in range(seq_len):
                p = torch.nn.functional.softmax(x[i, :, idx], dim=1)
                x_new = torch.zeros(p.size())
                x_new.scatter_(dim=1, index=torch.multinomial(p, 1), value=1)
                x[i, :, idx] = x_new
                log_likelihood += torch.sum(p * x_new)

        return x, log_likelihood

    def _from_input(self, x):
        """Transform from input space to dataframe.

        This maps categorical columns from a one-hot-like representation into
        the underlying categorical value.
        """
        if len(x.shape) == 3:
            assert x.shape[1] == 1, "batch size should be 1"
            x = x[:, 0, :]
        rows = []
        seq_len, input_size = x.shape
        for i in range(seq_len):
            row = {}

            for col_name, (mu_idx, var_idx) in self._continuous.items():
                row[col_name] = x[i, mu_idx].item()

            for col_name, indices in self._categorical.items():
                ml_value, max_x = None, float("-inf")
                for value, idx in indices.items():
                    if x[i, idx] > max_x:
                        max_x = x[i, idx]
                        ml_value = value
                row[col_name] = ml_value

            rows.append(row)
        return pd.DataFrame(rows, columns=self._column_names)


if __name__ == "__main__":
    from deepecho.benchmark import Simple3

    for model in [GammaModel()]:
        benchmark = Simple3()
        print(benchmark.evaluate(model))
