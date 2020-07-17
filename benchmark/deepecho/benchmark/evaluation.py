"""Top-level package for DeepEcho Benchmarking."""

import logging
from datetime import datetime

from deepecho.benchmark.dataset import Dataset

LOGGER = logging.getLogger(__name__)


def _evaluate_model_on_dataset(name, model, dataset, metrics):
    LOGGER.info('Evaluating model %s on %s', name, dataset)

    result = {
        'model': name,
        'dataset': str(dataset)
    }
    start = datetime.utcnow()

    try:
        if isinstance(dataset, str):
            dataset = Dataset(dataset)

        if isinstance(model, tuple):
            model_class, model_kwargs = model
            model = model_class(**model_kwargs)
        elif isinstance(model, type):
            model = model_class()

        model.fit(
            data=dataset.data,
            entity_columns=dataset.entity_columns,
            context_columns=dataset.context_columns
        )
        fit_end = datetime.utcnow()
        result['fit_time'] = (fit_end - start).total_seconds()

        context_columns = dataset.entity_columns + dataset.context_columns
        context = dataset.data[context_columns].drop_duplicates()
        sampled = model.sample(context=context)
        sample_end = datetime.utcnow()
        result['sample_time'] = (sample_end - fit_end).total_seconds()

        metric_start = sample_end
        for name, metric in metrics.items():
            try:
                score = metric(dataset, sampled)
                if isinstance(score, float):
                    result[name] = score
                elif isinstance(score, dict):
                    for key, value in score.items():
                        result['{}_{}'.format(name, key)] = value
                elif isinstance(score, tuple):
                    for i, value in enumerate(score):
                        result['{}_{}'.format(name, i)] = value

                metric_end = datetime.utcnow()
                result[name + '_time'] = (metric_end - metric_start).total_seconds()
                metric_start = metric_end
            except Exception:
                LOGGER.exception('Error running metric %s dataset %s',
                                 name, dataset.name)

    except Exception:
        LOGGER.exception('Error running model %s on dataset %s',
                         model_class.__name__, dataset.name)

    return result


def evaluate_model_on_datasets(name, model, datasets, metrics, distributed=False):
    """Evaluate the given model on a list of datasets.

    Args:
        model (class):
            Class of the model to evaluate or tuple containing the model
            class and the keyword arguments to use to initialize it.
        datasets (list):
            List of datasets in which to evaluate the model.
        metrics (dict):
            Dict of metrics to use for the evaluation.
        distributed (bool):
            Whether to use dask for distributed computing.
            Defaults to ``False``.

    Returns:
        pandas.DataFrame:
            Table containing the model name, the dataset name the scores
            obtained and the time elapsed during each stage for each one
            of the given datasets.
    """
    results = []

    if distributed:
        import dask
        function = dask.delayed(_evaluate_model_on_dataset)
    else:
        function = _evaluate_model_on_dataset

    for dataset in datasets:
        result = function(name, model, dataset, metrics)
        results.append(result)

    return results
