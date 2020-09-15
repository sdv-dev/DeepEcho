"""DeepEcho benchmarking setup. This file must be identical to deepecho/__init__.py."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.3.dev0'
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from deepecho.demo import load_demo
from deepecho.models.basic_gan import BasicGANModel
from deepecho.models.par import PARModel

__all__ = [
    'load_demo',
    'BasicGANModel',
    'PARModel',
]
