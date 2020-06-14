
class Model():
    """Base class for DeepEcho models.
    """

    @classmethod
    def check(cls, dataset):
        """Check whether the dataset can be modeled.

        Args:
            dataset: A `Dataset` object.

        Raises:
            ValueError: If the dataset is not supported by this model.
        """
        raise ValueError()

    def fit(self, dataset):
        """Fit the model to the dataset.

        Args:
            dataset: A `Dataset` object.
        """
        raise NotImplementedError()

    def sample(self):
        """Generate a synthetic dataset.

        Returns:
            A `Dataset` object.
        """
        raise NotImplementedError()
