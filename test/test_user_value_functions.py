from unittest import TestCase

from user_value_functions import LinearUserValueFunction, ChebycheffUserValueFunction


class TestLinearUserValueFunction(TestCase):
    def setUp(self):
        self.linear2 = LinearUserValueFunction([0.4, 0.6], False)
        self.linear3 = LinearUserValueFunction([0.3, 0.45, 0.25], False)
        self.linear4 = LinearUserValueFunction([0.3, 0.35, 0.25, 0.1], False)

    def test2(self):
        self.assertAlmostEqual(self.linear2.calculate([0.1, 0.5]), 0.34)

    def test3(self):
        self.assertAlmostEqual(self.linear3.calculate([0.25, 0.45, 0.9]), 0.5025)

    def test4(self):
        self.assertAlmostEqual(self.linear4.calculate([0.15, 0.68, 0.45, 0.81]), 0.4765)


class TestChebycheffUserValueFunction(TestCase):
    def setUp(self):
        self.chebycheff2 = ChebycheffUserValueFunction([0.4, 0.6], False)
        self.chebycheff3 = ChebycheffUserValueFunction([0.3, 0.45, 0.25], False)
        self.chebycheff4 = ChebycheffUserValueFunction([0.3, 0.35, 0.25, 0.1], False)

    def test2(self):
        self.assertAlmostEqual(self.chebycheff2.calculate([0.1, 0.5]), 0.3)

    def test3(self):
        self.assertAlmostEqual(self.chebycheff3.calculate([0.25, 0.45, 0.9]), 0.225)

    def test4(self):
        self.assertAlmostEqual(self.chebycheff4.calculate([0.15, 0.68, 0.45, 0.81]), 0.238)
