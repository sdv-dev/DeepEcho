"""The `model` module provides tools for modeling datasets.
"""


from .alpha import AlphaModel
from .base import Model
from .beta import BetaModel

__all__ = ['Model', 'AlphaModel', 'BetaModel']


def compatible_models(dataset):
    """Return a list of compatible model classes.

    Args:
        dataset: A `Dataset` object.

    Returns:
        A `list` of model classes than can be fitted to this dataset.
    """
    models = []
    for cls in Model.__subclasses__():
        if cls.check(dataset):
            models.append(cls)
    return models
