"""Integration tests for ``PARModel``."""

import unittest

import numpy as np

from deepecho.models.par import PARModel


class TestPARModel(unittest.TestCase):
    """Test class for the ``PARModel``."""

    def test_basic(self):
        """Test the basic usage of a ``PARModel``."""
        sequences = [
            {
                'context': [],
                'data': [
                    [0.0, np.nan, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                ]
            },
            {
                'context': [],
                'data': [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1, np.nan],
                ]
            }
        ]
        context_types = []
        data_types = ['continuous', 'continuous']

        model = PARModel()
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([])

    def test_conditional(self):
        """Test the ``PARModel`` with conditional sampling."""
        sequences = [
            {
                'context': [0],
                'data': [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, np.nan, 0.0],
                ]
            },
            {
                'context': [1],
                'data': [
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    [0.0, 0.1, np.nan, 0.3, 0.4, 0.5],
                ]
            }
        ]
        context_types = ['categorical']
        data_types = ['continuous', 'continuous']

        model = PARModel()
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])

    def test_mixed(self):
        """Test the ``PARModel`` with mixed input data."""
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
                    [0.5, np.nan, 0.3, 0.2, np.nan, 0.0],
                    [0, 1, 0, 1, np.nan, 1],
                ]
            }
        ]
        context_types = ['categorical']
        data_types = ['continuous', 'categorical']

        model = PARModel()
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])

    def test_count(self):
        """Test the PARModel with datatype ``count``."""
        sequences = [
            {
                'context': [0.5],
                'data': [
                    [0, 5, 5, np.nan, 1, 1],
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

        model = PARModel()
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])

    def test_variable_length(self):
        """Test ``PARModel`` with variable data length."""
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
                    [np.nan, 1, 0, 1, 0, np.nan],
                ]
            }
        ]
        context_types = ['count']
        data_types = ['count', 'categorical']

        model = PARModel()
        model.fit_sequences(sequences, context_types, data_types)
        model.sample_sequence([0])
