# DeepEcho Benchmark
This package contains code for benchmarking **DeepEcho** models. It takes as
input a set of `tasks` which corresponds to a dataset and associated task
description (described below) and reports the performance of a `model` instance
on the task.

## Usage

### CLI

```bash
$ deepecho-benchmark ~/deepecho_data
```

### Python API

```python
from deepecho_benchmark import run_benchmark

path_to_datasets = "~/deepecho_data"
results = run_benchmark(path_to_datasets)
print(results.head())
```

## Tasks
Each `task` in the benchmark is stored as a ZIP file on S3 which contains:

```
/<DATASET_NAME>
    <DATASET_NAME>.csv
    task.json
    metadata.json
    README.md
```

where `task.json` describes the task. Currently, there are only two types of
tasks; the task description tells the benchmark how to configure the model
and load the data from the CSV file.

## Simple Tasks
This type of task is intended to evaluate the performance of the model for
generating synthetic time series in general. For example, the task
description could look something like:

```json
{
    "task_type": "simple",
    "key": ["e_id"],
    "context": ["industry", "sector"],
}
```

The `key` field specifies the columns of the CSV file to group by in order
to separate the data into distinct entities. The `context` field specifies
the variables that are fixed for each entity and should be used to condition
the model.

## Classification Tasks
This type of task is intended to evaluate the performance of the model for
generating synthetic data for time series classification. For example, the task
description could look something like:

```json
{
    "task_type": "classification",
    "key": ["e_id"],
    "target": "ml_class",
    "ignored": ["tt_split", "s_index"]
}
```

The `key` field specifies the columns of the CSV file to group by in order
to separate the data into distinct entities. The `target` field specifies
the class label for each time series. The `ignored` field specifies the
columns to drop from the dataset.
