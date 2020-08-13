import unittest

import pandas as pd

from deepecho import PARModel


class TestDeepEcho(unittest.TestCase):

    def test_deepecho(self):
        model = PARModel()
        data = pd.DataFrame({
            'id': [0, 0, 0, 1, 1, 1],
            'x': [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        })

        model.fit(data, entity_columns=['id'])

        model.sample(5)
