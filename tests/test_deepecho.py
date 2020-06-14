import unittest

from deepecho.benchmark import Simple1, Simple2
from deepecho.model import AlphaModel


class TestDeepEcho(unittest.TestCase):

    def test_alpha_simple_1(self):
        benchmark = Simple1()
        benchmark.evaluate(AlphaModel())

    def test_alpha_simple_2(self):
        benchmark = Simple2()
        benchmark.evaluate(AlphaModel())
