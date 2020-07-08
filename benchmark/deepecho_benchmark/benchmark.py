import os
from collections import defaultdict
from glob import glob

import pandas as pd

from deepecho import PARModel
from deepecho_benchmark.tasks import Task


def run_benchmark(dataset_dir):
    """Evaluate the models on these datasets.

    Args:
        dataset_dir: The path to a directory containing one or more
            benchmark tasks.
    """
    task_type_to_results = defaultdict(list)
    for path_to_csv in glob(os.path.join(dataset_dir, "**/*.csv")):
        path_to_task = os.path.dirname(path_to_csv)
        task = Task.load(path_to_task)
        for model in [PARModel()]:
            results = task.evaluate(model)
            results["task"] = task.__class__.__name__
            results["model"] = model.__class__.__name__
            task_type_to_results[results["task"]].append(results)
    return {task_type: pd.DataFrame(rows) for task_type, rows in task_type_to_results.items()}
