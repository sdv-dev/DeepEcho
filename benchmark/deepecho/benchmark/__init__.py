"""Top-level package for DeepEcho Benchmarking."""

import logging
from datetime import datetime, timedelta

import pandas as pd

from deepecho import DeepEcho, PARModel
from deepecho.benchmark.dataset import Dataset, get_datasets_list
from deepecho.benchmark.evaluation import evaluate_model_on_datasets
from deepecho.benchmark.metrics import METRICS

__all__ = [
    'Dataset',
    'get_datasets_list',
    'run_benchmark'
]


DEFAULT_MODELS = {
    'PARModel': (PARModel, {'epochs': 256, 'cuda': True})
}


def _get_models_dict(models):
    if isinstance(models, dict):
        return models

    if models is None:
        return DEFAULT_MODELS

    models_dict = {}
    for model in models:
        if isinstance(model, str):
            models_dict[model] = DEFAULT_MODELS[model]
        elif isinstance(model, (tuple, list)):
            model, model_kwargs = model
            if isinstance(model, str):
                model = DEFAULT_MODELS[model][0]

            models_dict[model.__name__] = model, model_kwargs
        elif isinstance(type) and issubclass(model, DeepEcho):
            models_dict[model.__name__] = model
        else:
            TypeError('Invalid model type')

    return models_dict


def _get_metrics_dict(metrics):
    if isinstance(metrics, dict):
        return metrics

    if metrics is None:
        return METRICS

    return {
        metric: METRICS.get(metric) or METRICS[metric + '_score']
        for metric in metrics
    }


def progress(*futures):
    """Track progress of dask computation in a remote cluster.

    LogProgressBar is defined inside here to avoid having to import
    its dependencies if not used.
    """
    from distributed.client import futures_of
    from distributed.diagnostics.progressbar import TextProgressBar

    class LogProgressBar(TextProgressBar):
        last = 0
        logger = logging.getLogger('distributed')

        def _draw_bar(self, remaining, all, **kwargs):
            frac = (1 - remaining / all) if all else 0

            if frac > self.last + 0.01:
                self.last = int(frac * 100) / 100
                bar = "#" * int(self.width * frac)
                percent = int(100 * frac)

                time_per_task = self.elapsed / (all - remaining)
                remaining_time = timedelta(seconds=time_per_task * remaining)
                eta = datetime.utcnow() + remaining_time

                elapsed = timedelta(seconds=self.elapsed)
                msg = "[{0:<{1}}] | {2}% Completed | {3} | {4} | {5}".format(
                    bar, self.width, percent, elapsed, remaining_time, eta
                )
                self.logger.info(msg)

        def _draw_stop(self, **kwargs):
            pass

    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]

    LogProgressBar(futures)


def run_benchmark(models=None, datasets=None, metrics=None, max_entities=None,
                  distributed=False, output_path=None):
    """Score the indicated models on the indicated datasets.

    Args:
        models (list or dict):
            Models to evaluate, passed as a list of model names, or model
            classes or (model class, model kwargs) tuples, or as a dict of
            names and model classes or (model_class, model_kwargs) tuples.
            If a list of model names are passed, they are taken from the
            ``DEFAULT_MODELS`` dictionary.
            If not passed at all, the complete ``DEFAULT_MODELS`` dict is used.
        datasets (list):
            List of datasets in which to evaluate the model. They can be
            passed as dataset instances or as dataset names or paths.
            If not passed, all the available datasets are used.
        metrics (list):
            List of metric names to use.
            If not passed, all the available metrics are used.
        max_entities (int):
            Max number of entities to load per dataset.
            Defaults to ``None``.
        distributed (bool):
            Whether to use dask for distributed computing.
            Defaults to ``False``.
        output_path (str):
            If passed, store the results as a CSV in the given path.

    Returns:
        pandas.DataFrame:
            Table containing the model name, the dataset name the scores
            obtained and the time elapsed during each stage for each one
            of the given datasets and models.
    """
    models = _get_models_dict(models)
    metrics = _get_metrics_dict(metrics)

    if datasets is None:
        datasets = get_datasets_list()

    delayed = []
    for name, model in models.items():
        result = evaluate_model_on_datasets(
            name, model, datasets, metrics, max_entities, distributed)
        delayed.extend(result)

    if distributed:
        import dask
        persisted = dask.persist(*delayed)

        try:
            progress(persisted)
        except ValueError:
            # Using local client. No progress bar needed.
            pass

        results = dask.compute(*persisted)

    else:
        results = delayed

    results = pd.DataFrame(results)
    results = results[sorted(results.columns)]
    results.insert(0, 'model', results.pop('model'))
    results.insert(1, 'dataset', results.pop('dataset'))
    if output_path:
        results.to_csv(output_path, index=False)
    else:
        return results
