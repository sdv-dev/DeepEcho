import unittest

from deepecho.models.basic_gan import BasicGANModel


class TestBasicGANModel(unittest.TestCase):

    def test_basic(self):
        sequences = [
            {
                'context': [],
                'data': [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                ]
            },
            {
                'context': [],
                'data': [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                ]
            }
        ]
        context_types = []
        data_types = ['continuous', 'continuous']

        model = BasicGANModel(epochs=10)
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([])

    def test_conditional(self):
        sequences = [
            {
                'context': [0],
                'data': [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                ]
            },
            {
                'context': [1],
                'data': [
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                ]
            }
        ]
        context_types = ['categorical']
        data_types = ['continuous', 'continuous']

        model = BasicGANModel(epochs=10)
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])

    def test_mixed(self):
        sequences = [
            {
                'context': [0],
                'data': [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0, 1, 0, 1, 0, 1],
                ]
            },
            {
                'context': [1],
                'data': [
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    [0, 1, 0, 1, 0, 1],
                ]
            }
        ]
        context_types = ['categorical']
        data_types = ['continuous', 'categorical']

        model = BasicGANModel(epochs=10)
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])

    def test_count(self):
        sequences = [
            {
                'context': [0.5],
                'data': [
                    [0, 5, 5, 3, 1, 1],
                    [0, 1, 2, 1, 0, 1],
                ]
            },
            {
                'context': [1.1],
                'data': [
                    [1, 6, 6, 4, 2, 2],
                    [0, 1, 0, 1, 0, 1],
                ]
            }
        ]
        context_types = ['continuous']
        data_types = ['count', 'categorical']

        model = BasicGANModel(epochs=10)
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])

    def test_variable_length(self):
        sequences = [
            {
                'context': [0],
                'data': [
                    [0, 5, 5, 3, 1, 1, 0],
                    [0, 1, 2, 1, 0, 1, 2],
                ]
            },
            {
                'context': [1],
                'data': [
                    [1, 6, 6, 4, 2, 2],
                    [0, 1, 0, 1, 0, 1],
                ]
            }
        ]
        context_types = ['count']
        data_types = ['count', 'categorical']

        model = BasicGANModel(epochs=10)
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])
