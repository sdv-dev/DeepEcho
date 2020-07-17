"""DeepEcho Benchmarking metrics."""

from deepecho.benchmark.metrics.classification import classification_score, detection_score
from deepecho.benchmark.metrics.sdmetrics import sdmetrics_score

__all__ = [
    'sdmetrics_score',
    'classification_score',
    'detection_score'
]

METRICS = {
    'sdmetrics_score': sdmetrics_score,
    'classification_score': classification_score,
    'detection_score': detection_score,
}
