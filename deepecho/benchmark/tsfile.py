from sktime.utils.load_data import load_from_tsfile_to_dataframe

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
        # create synthetic_x, synthetic_y
        #     for each class in train_y, create dataset
        #     fit and sample
        # fit_and_score(train_x, train_y, test_x, test_y)
        # fit_and_score(synthetic_x, synthetic_y, test_x, test_y)
        raise NotImplementedError()

    def fit_and_score(self, train_x, train_y, test_x, test_y):
        raise NotImplementedError()
