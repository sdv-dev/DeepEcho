import os
from glob import glob

import pandas as pd

from deepecho.models import PARModel
from deepecho_benchmark.tasks import Task


def benchmark(path_to_csv):
    path_to_task = os.path.dirname(path_to_csv)
    task = Task.load(path_to_task)
    return task.evaluate(PARModel())


def run_benchmark(dataset_dir):
    rows = []
    for path_to_csv in glob(os.path.join(dataset_dir, "**/*.csv")):
        rows.append(benchmark(path_to_csv))
    df = pd.DataFrame(rows)
    print(df)
