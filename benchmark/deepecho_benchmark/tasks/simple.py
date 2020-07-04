from .base import Task


class SimpleTask(Task):

    def __init__(self, path_to_task):
        raise NotImplementedError()

    def evaluate(self, model):
        raise NotImplementedError()
