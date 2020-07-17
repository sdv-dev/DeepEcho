"""SDMetrics based metrics."""

import sdmetrics


def sdmetrics_score(dataset, synthetic):
    """Compare the real and the synthetic data using SDMetrics.

    The returned score is the overall score of the SDMetrics report.

    Args:
        dataset (Dataset):
            The real dataset.
        synthetic (DataFrame):
            The sampled data.

    Returns:
        float
    """
    real_tables = {dataset.table: dataset.data}
    synthetic_tables = {dataset.table: synthetic}
    return sdmetrics.evaluate(dataset.metadata, real_tables, synthetic_tables).overall()
