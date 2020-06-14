import numpy as np
import pandas as pd

from deepecho import Dataset
from deepecho.benchmark.base import Benchmark


class SimpleBenchmark(Benchmark):
    """SimpleBenchmark benchmark dataset.

    A SimpleBenchmark is one where the synthetically generated data
    can be evaluated by a simple scoring function which only takes
    as input a single synthetically generated Dataset instance.
    """

    def visualize(self):
        """Sample and visualize the benchmark dataset.

        Returns:
            (plt.Figure): A Figure object.
        """
        return self.sample().visualize()

    def evaluate(self, model):
        """Evaluate the model on this benchmark.

        Args:
            model: A `Model` object.

        Returns:
            A `dict` object containing various metrics.
        """
        model.fit(self.sample())
        return self.score(model.sample())

    def sample(self):
        """Create a Dataset instance.

        The Dataset instance can be either fixed (i.e. read from disk) or
        randomly generated according to some fixed distribution.

        Returns:
            A `Dataset` object containing a time-series dataset.
        """
        raise NotImplementedError()

    def score(self, dataset):
        """Evaluate the given Dataset instance.

        Args:
            dataset: A `Dataset` object containing a synthetically generated
                time-series dataset.

        Returns:
            A `dict` object containing various metrics.
        """
        raise NotImplementedError()


class Simple1(SimpleBenchmark):
    """Fixed-length time-series with continuous values.
    """

    def __init__(self, seq_length=32, nb_entities=10):
        self.seq_length = seq_length
        self.nb_entities = nb_entities
        self.exemplar_df = pd.DataFrame({
            "x": np.linspace(-1.0, 1.0, num=seq_length),
            "y": np.sin(np.linspace(-3.14, 3.14, num=seq_length)),
            "z": np.cos(np.linspace(-3.14, 3.14, num=seq_length)),
        })

    def sample(self):
        """Create an instance of the Simple1 dataset.

        This works by adding Gaussian noise to the exemplar dataset. The
        exemplar dataset contains three continuous columns: linear, sine,
        and cosine.

        Returns:
            A `Dataset` object containing a time-series dataset.
        """
        dataframes = []
        for entity_id in range(self.nb_entities):
            df = self.exemplar_df.copy()
            df["id"] = entity_id
            df["x"] += np.random.normal(size=self.seq_length, scale=0.1)
            df["y"] += np.random.normal(size=self.seq_length, scale=0.1)
            df["z"] += np.random.normal(size=self.seq_length, scale=0.1)
            dataframes.append(df)
        df = pd.concat(dataframes)
        return Dataset(df, entity_idx="id", fixed_length=self.seq_length)

    def score(self, dataset):
        """Evaluate the given Dataset instance.

        This returns the MSE between the synthetic dataset and the exemplar
        dataset.

        Args:
            dataset: A `Dataset` object containing a synthetically generated
                time-series dataset.

        Returns:
            A `dict` object containing various metrics.
        """
        mse = []
        for _, df in dataset.df.groupby(dataset.entity_idx):
            df = df.drop(dataset.entity_idx, axis=1)
            diff = df.values - self.exemplar_df.values
            mse.append(np.mean(diff**2))
        return {
            "mean-squared-error": np.mean(mse)
        }


class Simple2(SimpleBenchmark):
    """Fixed-length time-series with continuous and categorical values.
    """

    def __init__(self, seq_length=32, nb_entities=10):
        self.seq_length = seq_length
        self.nb_entities = nb_entities
        self.exemplar_df = pd.DataFrame({
            "x": np.linspace(-1.0, 1.0, num=seq_length),
            "y": np.sin(np.linspace(-3.14, 3.14, num=seq_length)),
            "z": (["yes", "no"] * seq_length)[:seq_length],
        })

    def sample(self):
        """Create an instance of the Simple2 dataset.

        This works by adding Gaussian noise to the numerical columns of
        the exemplar dataset and Bernoulli noise to the categorical
        column. The exemplar dataset contains two continuous columns
        containing the linear and sine functions as well as a categorical
        column which cycles between "yes" and "no".

        Returns:
            A `Dataset` object containing a time-series dataset.
        """
        dataframes = []
        for entity_id in range(self.nb_entities):
            df = self.exemplar_df.copy()
            df["id"] = entity_id
            df["x"] += np.random.normal(size=self.seq_length, scale=0.1)
            df["y"] += np.random.normal(size=self.seq_length, scale=0.1)

            def flip(value):
                if np.random.random() < 0.05:
                    return "yes" if value == "no" else "no"
                return value
            df["z"] = [flip(v) for v in df["z"].values]

            dataframes.append(df)
        df = pd.concat(dataframes)
        return Dataset(df, entity_idx="id", fixed_length=self.seq_length)

    def score(self, dataset):
        """Evaluate the given Dataset instance.

        This returns the MSE between the synthetic dataset and the exemplar
        dataset for the continuous columns and the accuracy on the categorical
        columns.

        Args:
            dataset: A `Dataset` object containing a synthetically generated
                time-series dataset.

        Returns:
            A `dict` object containing various metrics.
        """
        mse, acc = [], []
        for _, df in dataset.df.groupby(dataset.entity_idx):
            diff = df[["x", "y"]].values - self.exemplar_df[["x", "y"]].values
            mse.append(np.mean(diff**2))
            acc.append(np.mean(df["z"] == self.exemplar_df["z"]))
        return {
            "mean-squared-error": np.mean(mse),
            "accuracy": np.mean(acc)
        }


class Simple3(SimpleBenchmark):
    """Variable-length time-series with continuous values.
    """

    def __init__(self, nb_entities=10):
        self.nb_entities = nb_entities

    def sample(self):
        """Create an instance of the Simple3 dataset.

        This works by sampling the length of the time series from a
        Gaussian distribution with mean 10 and variance 1 (and casting
        it to an integer). Then, the `x` and `y` columns are populated
        with uniformly spread values going from 0.0-1.0 and 1.0-0.0
        respectively.

        Returns:
            A `Dataset` object containing a time-series dataset.
        """
        dataframes = []
        for entity_id in range(self.nb_entities):
            seq_len = int(np.random.normal(loc=10, scale=1))
            df = pd.DataFrame({
                "x": np.linspace(0.0, 1.0, num=seq_len),
                "y": np.linspace(0.0, 1.0, num=seq_len)
            })
            df["id"] = entity_id
            dataframes.append(df)
        df = pd.concat(dataframes)
        return Dataset(df, entity_idx="id")

    def score(self, dataset):
        """Evaluate the given Dataset instance.

        This returns the MSE between the synthetic dataset and the exemplar
        dataset for the continuous columns and the accuracy on the categorical
        columns.

        Args:
            dataset: A `Dataset` object containing a synthetically generated
                time-series dataset.

        Returns:
            A `dict` object containing various metrics.
        """
        mse, lengths = [], []
        for _, df in dataset.df.groupby(dataset.entity_idx):
            seq_len = len(df)
            target = pd.DataFrame({
                "x": np.linspace(0.0, 1.0, num=seq_len),
                "y": np.linspace(0.0, 1.0, num=seq_len)
            })
            diff = df[["x", "y"]].values - target[["x", "y"]].values
            mse.append(np.mean(diff**2))
            lengths.append(seq_len)
        return {
            "mean-squared-error": np.mean(mse),
            "|E[L]-\\hat{L}|": abs(np.mean(lengths) - 10.0)
        }
