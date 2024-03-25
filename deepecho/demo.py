"""Functions to load demo data."""

import os

import pandas as pd

_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def load_demo():
    """Load the demo DataFrame."""
    return pd.read_csv(os.path.join(_DATA_PATH, 'demo.csv'), parse_dates=['date'])
