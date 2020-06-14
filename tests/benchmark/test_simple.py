import unittest

from deepecho.benchmark.simple import Simple1, Simple2, Simple3


class TestSimple(unittest.TestCase):

    def test_simple_1(self):
        benchmark = Simple1()
        benchmark.sample()
        benchmark.score(benchmark.sample())

    def test_simple_2(self):
        benchmark = Simple2()
        benchmark.sample()
        benchmark.score(benchmark.sample())

    def test_simple_3(self):
        benchmark = Simple3()
        benchmark.sample()
        benchmark.score(benchmark.sample())
