"""Top-level package for DeepEcho Benchmarking."""

import logging
from datetime import datetime

import pandas as pd

from deepecho import DeepEcho, PARModel
from deepecho.benchmark.dataset import Dataset, get_datasets_list
from deepecho.benchmark.metrics import METRICS

LOGGER = logging.getLogger(__name__)


def _evaluate_model_on_dataset(model_class, model_kwargs, dataset, metrics):
    LOGGER.info('Evaluating model %s on %s', model_class.__name__, dataset)
    result = {
        'model': model_class.__name__,
        'dataset': str(dataset)
    }
    start = datetime.utcnow()

    try:
        if isinstance(dataset, str):
            dataset = Dataset(dataset)

        model = model_class(**model_kwargs)
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


def _evaluate_model_on_datasets(model_class, model_kwargs, datasets, metrics, distributed=False):
    results = []

    if distributed:
        import dask
        function = dask.delayed(_evaluate_model_on_dataset)
    else:
        function = _evaluate_model_on_dataset

    for dataset in datasets:
        result = function(model_class, model_kwargs, dataset, metrics)
        results.append(result)

    return results


DEFAULT_MODELS = {
    'PARModel': (PARModel, {'epochs': 256, 'cuda': True})
}


def benchmark(models=None, datasets=None, metrics=None, distributed=False):
    """Score the indicated models on the indicated datasets.

    Args:
        models (list):
            Models
        datasets (list):
            Datasets
        distributed (bool):
            Whether to use dask to distribute the computation.

    Returns:
        pandas.DataFrame
    """

    if models is None:
        models = DEFAULT_MODELS

    if datasets is None:
        datasets = get_datasets_list()

    if metrics is None:
        metrics = METRICS

    delayed = []
    for model in models:
        if isinstance(model, str):
            model, model_kwargs = DEFAULT_MODELS[model]
        elif isinstance(model, tuple):
            model, model_kwargs = model
        elif issubclass(model, DeepEcho):
            model_kwargs = {}
        else:
            TypeError('Invalid model type')

        result = _evaluate_model_on_datasets(model, model_kwargs, datasets, metrics, distributed)
        delayed.extend(result)

    if distributed:
        persisted = dask.persist(*delayed)
        results = dask.compute(*persisted)
    else:
        results = delayed

    return pd.DataFrame(results)
