"""DeepEcho Benchmarking Tasks module."""

from deepecho.benchmark.tasks.base import Task
from deepecho.benchmark.tasks.classification import ClassificationTask
from deepecho.benchmark.tasks.simple import SimpleTask

__all__ = [
    'Task',
    'ClassificationTask',
    'SimpleTask',
]
