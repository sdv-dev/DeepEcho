"""Integration tests for ``BasicGANModel``."""

import unittest

from deepecho.models.basic_gan import BasicGANModel


class TestBasicGANModel(unittest.TestCase):
    """Test class for the ``BasicGANModel``."""

    def test_basic(self):
        """Basic test for the ``BasicGANModel``."""
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
        """Test the ``BasicGANModel`` with conditional sampling."""
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
        """Test the ``BasicGANModel`` with mixed input data."""
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
        """Test the BasicGANModel with datatype ``count``."""
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
        """Test ``BasicGANModel`` with variable data length."""
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
