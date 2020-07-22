# Benchmarking DeepEcho

**DeepEcho** provides a benchmarking framework that allows users and developers to evaluate the
performance of the different models implemented in DeepEcho on a collection of real world
datasets.

## The Benchmarking process

The DeepEcho Benchmarking process has three main components:

### Datasets

We use the DeepEcho models to model and then sample a large collection of datasets of different
types. The collection of datasets can be found in the [deepecho-data bucket on S3](
http://deepecho-data.s3.amazonaws.com/index.html).

Most notably, many datasets from this collection are Time Series Classification datasets
downloaded from the [timeseriesclassification.com](http://www.timeseriesclassification.com/)
website.

### Modeling and Sampling process

During our benchmarking process, we use the DeepEcho models to learn the distributions of
these TimeSeries datasets conditioned on the contextual values associated with each entity
found in the datasets.

Afterwards, we generate synthetic data for each one of these entities, which generates a
dataset of synthetic time series with exactly the same size and aspect as the original data.

### Metrics

After modeling the Time Series datasets and then sampling synthetic data for each one of the
entities found within it, we apply several metrics the evaluate how similar the generated data
is to the real one.

We currently implement three metrics:

* SDMetrics Overall Score: We use [SDMetrics](/sdv-dev/SDMetrics) to generate a report and then
  obtain the overall score from it.
* Simple Detection Score: We fit a TimeSeriesForestClassifier from [sktime](https://sktime.org/)
  with a mixture of real and synthetic time series, indicating it which one is which. Later on
  we try to use the learned model to distinguish real and synthetic data from a held out partition.
* Classification Score: We fit a TimeSeriesForestClassifier from [sktime](https://sktime.org/)
  with real and synthetic time series independently. Afterwards, we use both models to evaluate
  accuracy on real held out data and then compare the obtained accuracies.

## Running the Benchmarking

### Install

Before running the benchmarking process, you will have to follow this two steps in order to
install the package:

#### Python installation

In order to use the DeepEcho benchmarking framework you will need to install it using the
following command:

```bash
pip install deepecho-benchmark
```

### Running the Benchmarking using python

The user API for the DeepEcho Benchmarking is the `deepecho.benchmark.run_benchmark` function.

The simplest usage is to execute the `run_benchmark` function without any arguments:

```python
from deepecho.benchmark import run_benchmark

scores = run_benchmark()
```

> :warning: Be aware that that this command takes a lot of time to run on a single machine!

This will execute all the DeepEcho models on all the available datasets and evaluate them
using all the metrics, producing a result similar to this one:

| model    | dataset            | fit_time | sample_time | detection_score | detection_score_time | sdmetrics_score | sdmetrics_score_time |
|----------|--------------------|----------|-------------|-----------------|----------------------|-----------------|----------------------|
| PARModel | Libras             | 664.417  |     381.354 |      0.0833333  |              33.5703 |       -0.457184 |             0.346504 |
| PARModel | AtrialFibrillation |  55.6314 |     779.799 |      0.333333   |              10.0637 |        0.236945 |             0.324723 |
| PARModel | BasicMotions       | 220.915  |     421.652 |      0.025      |              18.616  |       -2.39799  |             0.472108 |
| PARModel | ERing              | 599.311  |     632.197 |      0.00666667 |              33.5064 |       -4.08784  |             0.314248 |
| PARModel | RacketSports       | 720.873  |     277.63  |      0.0723684  |              32.3334 |       -1.91078  |             0.342887 |

Which contains:

* `model`: The name of the model that has been used.
* `dataset`: The name or path of the dataset.
* `fit_time`: Time, in seconds, that the training lasted.
* `sample_time`: Time, in seconds, that the sampling lasted.

And then, for each one of the metrics used:

* `<metric-name>`: Score obtained by the metric
* `<metric-name>_time`: Time, in seconds, that took to compute the metric.

### Benchmark Arguments

The `run_benchmark` function has the following optional arguments:

- `models`: List of models to evaluate, passed as classes or model
  names or as a tuples containing the class and the keyword
  arguments. If not passed, all the available models are used.
- `datasets`: List of datasets in which to evaluate the model. They can be
  passed as dataset instances or as dataset names or paths. If not passed,
  all the available datasets are used.
- `metrics`: Dict of metrics to use for the evaluation. If not passed, all the
  available metrics are used.
- `max_entities`: Max number of entities to load per dataset. If not given, use the
  entire dataset.
- `distributed`: Whether to use `dask` for distributed computing. Defaults to `False`.
- `output_path`: Optionally store the results in a CSV in the given path.


## Kubernetes

Running the complete DeepEcho Benchmarking suite can take a long time when executing against all
our datasts. For this reason, it comes prepared to be executed distributedly over a dask cluster
created using Kubernetes. Check our [documentation](KUBERNETES.md)
on how to run on a kubernetes cluster.
