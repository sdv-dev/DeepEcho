"""Top-level package for DeepEcho Benchmarking."""

import logging
from datetime import datetime

from deepecho.benchmark.dataset import Dataset

LOGGER = logging.getLogger(__name__)


def _log_time(result=None, name=None, last=None):
    now = datetime.utcnow()
    if last:
        result[name + '_time'] = now - last

    return now


def _fit_model(dataset, model, max_entities):
    if isinstance(dataset, str):
        dataset = Dataset(dataset, max_entities=max_entities)
    elif isinstance(dataset, list):
        dataset = Dataset(*dataset)

    if isinstance(model, tuple):
        model_instance = model[0](**model[1])
    elif isinstance(model, type):
        model_instance = model()

    model_instance.fit(
        data=dataset.data,
        entity_columns=dataset.entity_columns,
        context_columns=dataset.context_columns
    )

    return model_instance


def _sample(model_instance, dataset):
    context_columns = dataset.entity_columns + dataset.context_columns
    context = dataset.data[context_columns].drop_duplicates()
    return model_instance.sample(context=context)


def _compute_metric(dataset, sampled, metric_name, metric, result):
    score = metric(dataset, sampled)
    if isinstance(score, float):
        result[metric_name] = score
    elif isinstance(score, dict):
        for key, value in score.items():
            result['{}_{}'.format(metric_name, key)] = value
    elif isinstance(score, tuple):
        for i, value in enumerate(score):
            result['{}_{}'.format(metric_name, i)] = value


def _evaluate_model_on_dataset(model_name, model, dataset, metrics, max_entities=None):
    LOGGER.info('Evaluating model %s on %s', model_name, dataset)

    result = {
        'model': model_name,
        'dataset': str(dataset)
    }
    now = _log_time()

    try:
        model_instance = _fit_model(dataset, model, max_entities)
        now = _log_time(result, 'fit', now)

        sampled = _sample(model_instance, dataset)
        now = _log_time(result, 'sample', now)

        for metric_name, metric in metrics.items():
            try:
                _compute_metric(dataset, sampled, metric_name, metric, result)
                now = _log_time(result, metric_name, now)
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception('Error running metric %s dataset %s', metric_name, str(dataset))

    except Exception:  # pylint: disable=broad-except
        LOGGER.exception('Error running model %s on dataset %s', model_name, str(dataset))

    return result


def evaluate_model_on_datasets(name, model, datasets, metrics,
                               max_entities=None, distributed=False):
    """Evaluate the given model on a list of datasets.

    Args:
        model (class):
            Class of the model to evaluate or tuple containing the model
            class and the keyword arguments to use to initialize it.
        datasets (list):
            List of datasets in which to evaluate the model.
        metrics (dict):
            Dict of metrics to use for the evaluation.
        max_entities (int):
            Max number of entities to load per dataset.
            Defaults to ``None``.
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
        import dask  # pylint: disable=import-outside-toplevel
        function = dask.delayed(_evaluate_model_on_dataset)
    else:
        function = _evaluate_model_on_dataset

    for dataset in datasets:
        result = function(name, model, dataset, metrics, max_entities)
        results.append(result)

    return results
