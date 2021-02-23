"""Top-level package for DeepEcho."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.0.dev0'
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from deepecho.demo import load_demo
from deepecho.models.basic_gan import BasicGANModel
from deepecho.models.par import PARModel

__all__ = [
    'load_demo',
    'BasicGANModel',
    'PARModel',
]
