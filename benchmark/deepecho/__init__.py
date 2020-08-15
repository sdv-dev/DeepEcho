"""DeepEcho benchmarking setup. This file must be identical to deepecho/__init__.py."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.1'
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from deepecho.base import DeepEcho
from deepecho.par import PARModel

__all__ = ['DeepEcho', 'PARModel']
