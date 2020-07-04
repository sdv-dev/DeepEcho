import os

from .benchmark import run_benchmark
from .download import download


def main():
    dataset_dir = os.path.expanduser("~/deepecho_data")
    download(dataset_dir)
    run_benchmark(dataset_dir)


if __name__ == "__main__":
    main()
