from deepecho.benchmark import run_benchmark
from deepecho.benchmark.dataset import Dataset


def test_run_benchmark():
    models = [('PARModel', {'epochs': 5, 'cuda': False, 'verbose': False})]
    datasets = [Dataset('Libras', max_entities=10)]
    metrics = ['sdmetrics', 'detection']
    results = run_benchmark(
        models=models,
        datasets=datasets,
        metrics=metrics
    )
