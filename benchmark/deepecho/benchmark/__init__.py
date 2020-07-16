"""Top-level package for DeepEcho Benchmarking."""

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


def run_benchmark(models=None, datasets=None, metrics=None, distributed=False, output_path=None):
    """Score the indicated models on the indicated datasets.

    Args:
        models (list):
            List of models to evaluate, passed as classes or model
            names or as a tuples containing the class and the keyword
            arguments.
            If not passed, the ``DEFAULT_MODELS`` are used.
        datasets (list):
            List of datasets in which to evaluate the model. They can be
            passed as dataset instances or as dataset names or paths.
            If not passed, all the available datasets are used.
        metrics (dict):
            Dict of metrics to use for the evaluation.
            If not passed, all the available metrics are used.
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

        result = evaluate_model_on_datasets(model, model_kwargs, datasets, metrics, distributed)
        delayed.extend(result)

    if distributed:
        import dask
        persisted = dask.persist(*delayed)
        results = dask.compute(*persisted)
    else:
        results = delayed

    results = pd.DataFrame(results)
    if output_path:
        results.to_csv(output_path, index=False)
    else:
        return results
