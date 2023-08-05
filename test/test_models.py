from unittest import TestCase
import numpy as np

from models import MostDiscriminatingValueFunction


class TestMostDiscriminatingValueFunction(TestCase):
    def setUp(self):
        self.model = MostDiscriminatingValueFunction(50, 4, True)
        self.model.best_obj = np.ones(4)
        self.model.worst_obj = np.full(4, fill_value=4.0)

    # test case 1 based on tables I & III from paper:
    # J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015
    def test_mdvf1(self):
        comparisons = [[np.array([2.0, 4.0, 4.0, 2.0]), np.array([4.0, 2.0, 2.0, 4.0])],  # a4 > a1
                       [np.array([4.0, 2.0, 2.0, 4.0]), np.array([3.0, 1.0, 3.0, 1.0])],  # a1 > a2
                       [np.array([2.0, 4.0, 4.0, 2.0]), np.array([3.0, 1.0, 3.0, 1.0])]]  # a4 > a2

        for pair in comparisons:
            self.model.update(pair)
        print(*self.model.interp_points, sep='\n')

        self.assertEqual(len(self.model.buffer), 3)  # check if model buffered all 3 comparisons
        self.assertAlmostEqual(self.model.value_funtion(np.array([4.0, 2.0, 2.0, 4.0])), 1.0 / 3.0)  # a1
        self.assertAlmostEqual(self.model.value_funtion(np.array([3.0, 1.0, 3.0, 1.0])), 0.0)   # a2
        self.assertAlmostEqual(self.model.value_funtion(np.array([1.0, 3.0, 1.0, 3.0])), 1.0)   # a3
        self.assertAlmostEqual(self.model.value_funtion(np.array([2.0, 4.0, 4.0, 2.0])), 2.0 / 3.0)  # a4

    # test case based on tables I & IX from paper:
    # J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015
    def test_mdvf2(self):
        comparisons = [[np.array([2.0, 4.0, 4.0, 2.0]), np.array([4.0, 2.0, 2.0, 4.0])],  # a4 > a1
                       [np.array([4.0, 2.0, 2.0, 4.0]), np.array([3.0, 1.0, 3.0, 1.0])],  # a1 > a2
                       [np.array([2.0, 4.0, 4.0, 2.0]), np.array([3.0, 1.0, 3.0, 1.0])],  # a4 > a2
                       [np.array([2.0, 4.0, 4.0, 2.0]), np.array([1.0, 3.0, 1.0, 3.0])]]  # a4 > a3

        for pair in comparisons:
            self.model.update(pair)
        print(*self.model.interp_points, sep='\n')

        # print(*self.model.interp_points, sep='\n')
        self.assertAlmostEqual(len(self.model.buffer), 2)  # check if model discarded 2 incompatible comparisons
        self.assertAlmostEqual(self.model.value_funtion(np.array([4.0, 2.0, 2.0, 4.0])), 0.0)  # a1
        self.assertAlmostEqual(self.model.value_funtion(np.array([3.0, 1.0, 3.0, 1.0])), 0.5)  # a2
        self.assertAlmostEqual(self.model.value_funtion(np.array([1.0, 3.0, 1.0, 3.0])), 0.5)  # a3
        self.assertAlmostEqual(self.model.value_funtion(np.array([2.0, 4.0, 4.0, 2.0])), 1.0)  # a4
