import json
import os


class Task():

    @classmethod
    def load(cls, path_to_task):
        """Load the task from disk.

        Args:
            path_to_task: The path to the task directory.

        Return:
            A Task instance.
        """
        from .simple import SimpleTask
        from .classification import ClassificationTask

        task_types = {
            "simple": SimpleTask,
            "classification": ClassificationTask
        }

        with open(os.path.join(path_to_task, "task.json"), "rt") as fin:
            task = json.load(fin)
        task_cls = task_types.get(task["task_type"], None)

        if not task_cls:
            raise ValueError("Unknown task type.")
        return task_cls(path_to_task)

    def evaluate(self, model):
        """Evaluate the model on this task.

        Args:
            model: An instance of a DeepEcho model.

        Return:
            A dictionary mapping metric names to values. The set of metrics
            that are returned is fixed for each subclass (i.e. each subclass
            will always return the same set of metrics).
        """
        raise NotImplementedError()
