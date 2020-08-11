"""DeepEcho Benchmarking metrics."""

from deepecho.benchmark.metrics.classification import (
    classification_score, lstm_detection_score, rf_detection_score)
from deepecho.benchmark.metrics.sdmetrics import sdmetrics_score

__all__ = [
    'sdmetrics_score',
    'classification_score',
    'lstm_detection_score',
    'rf_detection_score',
]

METRICS = {
    'sdmetrics_score': sdmetrics_score,
    'classification_score': classification_score,
    'lstm_detection_score': lstm_detection_score,
    'rf_detection_score': rf_detection_score,
}
