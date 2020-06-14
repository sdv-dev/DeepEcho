class Benchmark():
    """This represents a "template" for a benchmark dataset.

    Each `Benchmark` is a template for a dataset. To train a model, a concrete
    instance of the dataset can be retrieved using the `sample` method. Once
    the model is trained, it can be used to generate a synthetic dataset. This
    can then be passed to the `evaluate` method which evaluates how well the
    synthetically generated dataset conforms to the template.
    """

    def visualize(self):
        """Visualize the benchmark dataset.

        This function generates an instance of the dataset and plots it using
        matplotlib. It returns a Figure which can be shown and/or saved.

        Returns:
            (plt.Figure): A Figure object.
        """
        raise NotImplementedError()

    def evaluate(self, model):
        """Evaluate the model on this benchmark.

        Args:
            model: A `Model` object.

        Returns:
            A `dict` object containing various metrics.
        """
        raise NotImplementedError()
