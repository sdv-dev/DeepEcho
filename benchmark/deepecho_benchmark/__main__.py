import os

from .benchmark import run_benchmark
from .download import download


def main():
    dataset_dir = os.path.expanduser("~/deepecho_data")
    download(dataset_dir)
    for task_type, df in run_benchmark(dataset_dir).items():
        df.to_csv("%s.csv" % task_type, index=False)


if __name__ == "__main__":
    main()
