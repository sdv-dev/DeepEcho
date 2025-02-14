"""Top-level package for DeepEcho."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '0.7.0'
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from deepecho.demo import load_demo
from deepecho.models.basic_gan import BasicGANModel
from deepecho.models.par import PARModel

__all__ = [
    'load_demo',
    'BasicGANModel',
    'PARModel',
]
