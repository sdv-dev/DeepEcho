"""The `model` module provides tools for modeling datasets.
"""


from .alpha import AlphaModel
from .base import Model
from .beta import BetaModel
from .gamma import GammaModel

__all__ = [
    'Model',
    'AlphaModel',
    'BetaModel',
    'GammaModel',
]
