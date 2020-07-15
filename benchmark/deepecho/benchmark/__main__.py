"""Command Line interface for the DeepEcho Benchmark."""

import os

from deepecho.benchmark import run_benchmark
from deepecho.benchmark.download import download


def main():
    """Run the DeepEcho benchmark."""
    dataset_dir = os.path.expanduser('~/deepecho_data')
    download(dataset_dir)
    for task_type, df in run_benchmark(dataset_dir).items():
        df.to_csv('{}.csv'.format(task_type), index=False)


if __name__ == '__main__':
    main()
