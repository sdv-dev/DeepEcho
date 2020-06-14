import numpy as np
import pandas as pd
import torch

from deepecho import Dataset
from deepecho.model.base import Model


class Alpha1(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(Alpha1, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.start_token = torch.randn(input_size, requires_grad=True)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 32, kernel_size=3),
            torch.nn.Tanh(),
            torch.nn.Conv1d(32, output_size, kernel_size=3)
        )

        self.window_size = self._window_size()

    def _window_size(self):
        for window_size in range(32):
            x = self.start_token.unsqueeze(0).unsqueeze(2)
            x = x.expand(1, self.input_size, window_size)
            try:
                self.cnn(x)
                return window_size
            except RuntimeError:
                pass

    def forward(self, x):
        """Predict the next step in the sequence.

        This takes (N, C_in, L) and produces a new sequence (N, C_out, L) where
        the (:, :, i)th output is a function of the (:, :, :i-1)th inputs.
        """
        batch_size, input_size, sequence_length = x.shape

        prefix = self.start_token.unsqueeze(0).unsqueeze(2)
        prefix = prefix.expand(batch_size, self.input_size, self.window_size)

        y = self.cnn(torch.cat([prefix, x], dim=2))[:, :, :-1]
        assert x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2]
        return y

    def sample(self, x=False):
        """Generate the next step in the sequence.

        This optionally takes (N, C_in, 1) and produces a new element (N, C_out, 1).
        """
        if isinstance(x, bool):
            batch_size = 1
            prefix = self.start_token.unsqueeze(0).unsqueeze(2)
            prefix = prefix.expand(batch_size, self.input_size, self.window_size)
            y = self.cnn(prefix)
        else:
            batch_size, input_size, sequence_length = x.shape
            prefix = self.start_token.unsqueeze(0).unsqueeze(2)
            prefix = prefix.expand(batch_size, self.input_size, self.window_size)
            y = self.cnn(torch.cat([prefix, x], dim=2))
        return y[:, :, -1:]


class AlphaModel(Model):
    """No time index, multiple entities, fixed length.
    """

    def __init__(self, nb_epochs=128, use_sampling=True):
        self.nb_epochs = nb_epochs
        self.use_sampling = use_sampling

    @classmethod
    def check(cls, dataset):
        if dataset.time_idx:
            raise ValueError("This model doesn't support time indices.")
        if not dataset.entity_idx:
            raise ValueError("This model requires multiple entiies.")
        if not dataset.fixed_length:
            raise ValueError("This model requires a fixed length.")

    def fit(self, dataset):
        self.check(dataset)

        # Transform dataset into input space.
        X = []
        self._build(dataset)
        for _, df in dataset.df.groupby(dataset.entity_idx):
            df = df.drop(dataset.entity_idx, axis=1)
            X.append(self._from_df(df))
        X = torch.stack(X, dim=0)  # (N, C_in, L)

        self._model = Alpha1(self._input_size, self._input_size)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        for epoch in range(self.nb_epochs):
            Y = self._model(X)  # (N, C_in, L) -> (N, C_out, L)

            loss = 0.0

            # MSE for continuous columns
            continuous_idx = list(self._continuous.values())
            loss += torch.nn.functional.mse_loss(Y[:, continuous_idx, :], X[:, continuous_idx, :])

            # Cross entropy for categorical columns
            for _, indices in self._categorical.items():
                idx = list(indices.values())
                predicted, target = Y[:, idx, :], X[:, idx, :]
                target = torch.argmax(target, dim=1)
                loss += torch.nn.functional.cross_entropy(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def sample(self):
        dataframes = []

        for entity_id in range(self._nb_entities):
            x = self._from_latent(self._model.sample())
            for _ in range(self._seq_length - 1):
                next_x = self._from_latent(self._model.sample(x))
                x = torch.cat([x, next_x], dim=2)
            df = self._from_input(x)
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
                self._continuous[column] = idx
                idx += 1
            elif df[column].dtype == np.object:
                self._categorical[column] = {}
                for value in set(df[column]):
                    self._categorical[column][value] = idx
                    idx += 1
            else:
                raise ValueError("Unsupported type.")
        self._input_size = idx

    def _from_df(self, df):
        """Transform from dataframe to input space.

        This maps categorical columns from into a one-hot-like representation.
        """
        X = []
        for _, row in df.iterrows():
            x = torch.zeros(self._input_size)
            for key, value in row.items():
                if key in self._continuous:
                    x[self._continuous[key]] = value
                elif key in self._categorical:
                    x[self._categorical[key][value]] = 1.0
                else:
                    raise ValueError("Unknown column %s" % key)
            X.append(x)
        return torch.stack(X, dim=1)  # (C_in, L)

    def _from_latent(self, x):
        """Transform from latent space to input space.

        This maps categorical columns from the latent representation into a
        one-hot-like representation. If `use_sampling` is True, then the
        categorical values are sampled from the distribution as opposed to
        simply choosing the maximum likelihood value.
        """
        if self.use_sampling:
            batch_size, input_size, seq_len = x.shape
            for col_name, indices in self._categorical.items():
                idx = list(indices.values())
                for i in range(seq_len):
                    p = torch.nn.functional.softmax(x[:, idx, i], dim=1)
                    x_new = torch.zeros(p.size())
                    x_new.scatter_(dim=1, index=torch.multinomial(p, 1), value=1)
                    x[:, idx, i] = x_new
        return x

    def _from_input(self, x):
        """Transform from input space to dataframe.

        This maps categorical columns from a one-hot-like representation into
        the underlying categorical value.
        """
        if len(x.shape) == 3:
            assert x.shape[0] == 1, "batch size should be 1"
            x = x[0, :, :]
        rows = []
        input_size, seq_len = x.shape
        for i in range(seq_len):
            row = {}

            for col_name, idx in self._continuous.items():
                row[col_name] = x[idx, i].item()

            for col_name, indices in self._categorical.items():
                ml_value, max_x = None, float("-inf")
                for value, idx in indices.items():
                    if x[idx, i] > max_x:
                        max_x = x[idx, i]
                        ml_value = value
                row[col_name] = ml_value

            rows.append(row)
        return pd.DataFrame(rows, columns=self._column_names)
