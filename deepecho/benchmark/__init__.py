"""The `benchmark` module provides tools for sampling and evaluating datasets.
"""


from .base import Benchmark
from .simple import Simple1, Simple2, Simple3
from .tsfile import TSFileBenchmark

__all__ = [
    'Benchmark',
    'Simple1',
    'Simple2',
    'Simple3',
    'TSFileBenchmark'
]
