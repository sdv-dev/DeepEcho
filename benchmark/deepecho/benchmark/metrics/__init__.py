"""DeepEcho Benchmarking metrics."""

from deepecho.benchmark.metrics.classification import (
    real_vs_synthetic_score, simple_detection_score)
from deepecho.benchmark.metrics.sdmetrics import sdmetrics_overall

__all__ = [
    'sdmetrics_overall',
    'real_vs_synthetic_score',
    'simple_detection_score'
]

METRICS = {
    'sdmetrics_overall': sdmetrics_overall,
    'real_vs_synthetic_score': real_vs_synthetic_score,
    'simple_detection_score': simple_detection_score,
}
