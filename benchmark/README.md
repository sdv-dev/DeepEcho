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

This is the complete list of available datasets and some of their characteristics can be seen
in the `Datasets` tab of the [DeepEcho Benchmark Results spreadsheet on Google Drive](
https://docs.google.com/spreadsheets/d/1aCbdjOHD12l08NDfSRavFNvptJVqgLTF/)

Further details more details about how the format in which these datasets are stored as well
as how to create yours, please [follow this tutorial](../tutorials/02_DeepEcho_Benchmark_Datasets.ipynb)

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

We currently implement four metrics:

* SDMetrics Overall Score: We use [SDMetrics](/sdv-dev/SDMetrics) to generate a report and then
  obtain the overall score from it. A larger score indicates that the synthetic data is higher
  quality.

* Random Forest Detection Score: We fit a TimeSeriesForestClassifier from [sktime](https://sktime.org/)
  with a mixture of real and synthetic time series, indicating it which one is which. Later on
  we try to use the learned model to distinguish real and synthetic data from a held out partition.

* LSTM Detection Score: We train a LSTM classifier to distinguish between real and synthetic time
  series. We evaluate the performance of the classifier on a held out partition and report the
  error rate (i.e. larger values indicate that the synthetic data is higher quality).

* Classification Score: We fit a TimeSeriesForestClassifier from [sktime](https://sktime.org/)
  with real and synthetic time series independently. Afterwards, we use both models to evaluate
  accuracy on real held out data and report the ratio between the performance of the synthetic
  model and the performance of the real model (i.e. larger values indicate that the synthetic
  data is higher quality).

## Benchmark Results

For every release we run the DeepEcho Becnhmark on all our models and datasets to produce a
comprehensive table of results. These are the results obtained by the latest version of DeepEcho
using the following configuration:

- Models: `PARModel`
- Datasets: 25
- Maximum Entities: 1000

| model    | dataset                   | fit_time   | sample_time   |   classification_score | classification_score_time   |   lstm_detection_score | lstm_detection_score_time   |   rf_detection_score | rf_detection_score_time   |   sdmetrics_score | sdmetrics_score_time   |
|----------|---------------------------|------------|---------------|------------------------|-----------------------------|------------------------|-----------------------------|----------------------|---------------------------|-------------------|------------------------|
| PARModel | Libras                    | 11m12s     | 35s           |               0.3875   | 27s                         |             0.138889   | 6s                          |           0.105556   | 24s                       |         0.0473328 | 0s                     |
| PARModel | AtrialFibrillation        | 1m25s      | 1m21s         |               1        | 10s                         |             0.466667   | 18s                         |           0.133333   | 8s                        |         1.1935    | 0s                     |
| PARModel | BasicMotions              | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | ERing                     | 5m2s       | 1m26s         |               0.430556 | 35s                         |             0.08       | 7s                          |           0          | 32s                       |        -4.05752   | 0s                     |
| PARModel | RacketSports              | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | Epilepsy                  | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | PenDigits                 | 17m24s     | 25s           |               0.382353 | 36s                         |             0.008      | 7s                          |           0.034      | 32s                       |        -0.164203  | 0s                     |
| PARModel | JapaneseVowels            | 1h51m3s    | 1m20s         |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | StandWalkJump             | 2m32s      | 10m36s        |             inf        | 28s                         |             0.285714   | 50s                         |           0.0714286  | 25s                       |        -4.09482   | 1s                     |
| PARModel | FingerMovements           | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | EchoNASDAQ                | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | Handwriting               | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | UWaveGestureLibrary       | 25m30s     | 11m47s        |               0.30303  | 1m42s                       |             0.0227273  | 42s                         |           0.00454545 | 1m32s                     |        -3.44902   | 3s                     |
| PARModel | NATOPS                    | 2h2m52s    | 4m29s         |               0.536585 | 1m36s                       |             0.0222222  | 12s                         |           0.0222222  | 1m31s                     |      -262.196     | 3s                     |
| PARModel | ArticularyWordRecognition | 1h2m5s     | 7m9s          |               0.561151 | 2m20s                       |             0.104167   | 17s                         |           0          | 1m59s                     |       -26.5596    | 2s                     |
| PARModel | Cricket                   | 19m37s     | 1h31m51s      |               0.888889 | 3m57s                       |             0.0222222  | 1m9s                        |           0.0111111  | 3m45s                     |        -6.84794   | 3s                     |
| PARModel | SelfRegulationSCP2        | 18m60s     | 2h47m18s      |               1.01961  | 7m20s                       |             0.0105263  | 1m33s                       |           0.0105263  | 7m11s                     |       -14.7521    | 7s                     |
| PARModel | LSST                      | 56m12s     | 3m43s         |               0.833333 | 1m49s                       |             0.07       | 14s                         |           0.03       | 1m36s                     |        -0.295678  | 4s                     |
| PARModel | SelfRegulationSCP1        | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | CharacterTrajectories     | 46m17s     | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | HandMovementDirection     | 15m45s     | 33m29s        |               1.2381   | 3m26s                       |             0.264957   | 31s                         |           0.017094   | 3m16s                     |       -14.635     | 4s                     |
| PARModel | EthanolConcentration      | 19m48s     | 3h19m6s       |               1.17188  | 5m58s                       |             0.00381679 | 2m48s                       |           0          | 5m34s                     |        -4.11028   | 8s                     |
| PARModel | SpokenArabicDigits        | nan        | nan           |             nan        | nan                         |           nan          | nan                         |         nan          | nan                       |       nan         | nan                    |
| PARModel | Heartbeat                 | 5h24m41s   | 1h23m25s      |               1.25316  | 15m10s                      |             0.0146341  | 45s                         |           0.00487805 | 15m6s                     |       -36.6502    | 1m5s                   |
| PARModel | PhonemeSpectra            | 2h45m2s    | 28m31s        |               4.86957  | 7m55s                       |             0.018      | 53s                         |           0.042      | 6m2s                      |       -48.8024    | 6s                     |

Which include:

* `model`: The name of the model that has been used.
* `dataset`: The name or path of the dataset.
* `fit_time`: Time taken by the training process.
* `sample_time`: Time taken by the sampling process.

And then, for each one of the metrics used:

* `<metric-name>`: Score obtained by the metric.
* `<metric-name>_time`: Time, in seconds, that took to compute the metric.

Please, find the complete table of results for every release, as well as a summary of all the
available datasets, in the [DeepEcho Benchmark Results spreadsheet on Google Drive](
https://docs.google.com/spreadsheets/d/1aCbdjOHD12l08NDfSRavFNvptJVqgLTF/)

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
using all the metrics, producing a table similar to the one shown above.

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
