import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime.utils.load_data import load_from_tsfile_to_dataframe

from deepecho import Dataset
from deepecho.benchmark.base import Benchmark


class TSFileBenchmark(Benchmark):
    """Time series classification benchmark (*.ts files).

    This class loads *.ts files which represent time series classification
    problems. The *.ts file format is descibed at [1] and datasets stored
    using this format can be found at [2].

    For each class label, this fits a model to the training examples and
    uses it to generate a set of synthetic time series. It then uses those
    synthetic time series to train a simple time series classifier and reports
    the gap in performance beween a classifier trained on the real data and
    on the synthetic data.

    [1] https://alan-turing-institute.github.io/sktime/examples/loading_data.html

    [2] http://www.timeseriesclassification.com/dataset.php
    """

    def __init__(self, train_path, test_path):
        """Create a time series classification benchmark.

        Args:
            train_path: Path to the *.ts file for training.
            test_path: Path to the *.ts file for testing.
        """
        self.train_x, self.train_y = load_from_tsfile_to_dataframe(train_path)
        self.test_x, self.test_y = load_from_tsfile_to_dataframe(test_path)

    def evaluate(self, model):
        """Evaluate the model on this benchmark.

        Args:
            model: A `Model` object.

        Returns:
            A `dict` object containing various metrics.
        """
        # SKTime -> DataFrame
        label_to_dfs = {}
        for (idx, x), y in zip(self.train_x.iterrows(), self.train_y):
            df = pd.DataFrame()
            for col_name, col_value in x.iteritems():
                df[col_name] = col_value
            df["entity_id"] = idx
            if y not in label_to_dfs:
                label_to_dfs[y] = []
            label_to_dfs[y].append(df)

        # DataFrame -> Dataset
        label_to_dataset = {}
        for label, dfs in list(label_to_dfs.items()):
            label_to_dataset[label] = Dataset(
                df=pd.concat(dfs, axis=0),
                entity_idx="entity_id"
            )
            model.fit(label_to_dataset[label])
            label_to_dataset[label] = model.sample()

        # Dataset -> SKTime
        synthetic_x, synthetic_y = [], []
        for label, dataset in label_to_dataset.items():
            for _, df in dataset.df.groupby("entity_id"):
                df = df.drop("entity_id", axis=1)
                x = {}
                for column in df.columns:
                    x[column] = df[column]
                synthetic_x.append(x)
                synthetic_y.append(label)
        synthetic_x = pd.DataFrame(synthetic_x, columns=self.train_x.columns)
        synthetic_y = np.array(synthetic_y)

        real_acc = self.fit_and_score(self.train_x, self.train_y, self.test_x, self.test_y)
        synthetic_acc = self.fit_and_score(synthetic_x, synthetic_y, self.test_x, self.test_y)
        return {
            "synthetic_acc": synthetic_acc,
            "synthetic_acc-real_acc": synthetic_acc - real_acc,
        }

    def fit_and_score(self, train_x, train_y, test_x, test_y):
        # TODO: switch to a model that supports variable length sequences
        clf = Pipeline([
            ('concatenate', ColumnConcatenator()),
            ('classify', TimeSeriesForestClassifier())
        ])
        clf.fit(train_x, train_y)
        return clf.score(test_x, test_y)


if __name__ == "__main__":
    from deepecho.model.gamma import GammaModel
    benchmark = TSFileBenchmark(
        "/Users/kevz/Downloads/AllGestureWiimoteZ/AllGestureWiimoteZ_TRAIN.ts",
        "/Users/kevz/Downloads/AllGestureWiimoteZ/AllGestureWiimoteZ_TEST.ts",
    )
    print(benchmark.evaluate(GammaModel()))
