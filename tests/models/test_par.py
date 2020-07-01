import unittest

from deepecho.models import PARModel


class TestPARModel(unittest.TestCase):

    def test_basic(self):
        sequences = [
            {
                "context": [],
                "data": [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                ]
            },
            {
                "context": [],
                "data": [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                ]
            }
        ]
        context_types = []
        data_types = ["continuous", "continuous"]

        model = PARModel()
        model.fit(sequences, context_types, data_types)
        model.sample([])

    def test_conditional(self):
        sequences = [
            {
                "context": [0],
                "data": [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                ]
            },
            {
                "context": [1],
                "data": [
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                ]
            }
        ]
        context_types = ["categorical"]
        data_types = ["continuous", "continuous"]

        model = PARModel()
        model.fit(sequences, context_types, data_types)
        model.sample([0])

    def test_mixed(self):
        sequences = [
            {
                "context": [0],
                "data": [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0, 1, 0, 1, 0, 1],
                ]
            },
            {
                "context": [1],
                "data": [
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    [0, 1, 0, 1, 0, 1],
                ]
            }
        ]
        context_types = ["categorical"]
        data_types = ["continuous", "categorical"]

        model = PARModel()
        model.fit(sequences, context_types, data_types)
        model.sample([0])
