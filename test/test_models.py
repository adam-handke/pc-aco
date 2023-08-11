from unittest import TestCase
import numpy as np

from models import MostDiscriminatingValueFunction, MaximalSumOfScoresValueFunction


class TestMostDiscriminatingValueFunction(TestCase):
    # test case 1 based on tables I & III from paper:
    # J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015
    def test_mdvf1(self):
        print('Starting MDVF test 1')
        model = MostDiscriminatingValueFunction(30, 4, True,
                                                best_obj=np.ones(4), worst_obj=np.full(4, fill_value=4.0))
        comparisons = [[np.array([2.0, 4.0, 4.0, 2.0]), np.array([4.0, 2.0, 2.0, 4.0])],  # a4 > a1
                       [np.array([4.0, 2.0, 2.0, 4.0]), np.array([3.0, 1.0, 3.0, 1.0])],  # a1 > a2
                       [np.array([2.0, 4.0, 4.0, 2.0]), np.array([3.0, 1.0, 3.0, 1.0])]]  # a4 > a2

        for pair in comparisons:
            model.update(pair)
        print(*model.interp_points, sep='\n')

        self.assertEqual(len(model.buffer), 3)  # check if model buffered all 3 comparisons
        self.assertAlmostEqual(model.value_funtion(np.array([4.0, 2.0, 2.0, 4.0])), 1.0 / 3.0)  # a1
        self.assertAlmostEqual(model.value_funtion(np.array([3.0, 1.0, 3.0, 1.0])), 0.0)   # a2
        self.assertAlmostEqual(model.value_funtion(np.array([1.0, 3.0, 1.0, 3.0])), 1.0)   # a3
        self.assertAlmostEqual(model.value_funtion(np.array([2.0, 4.0, 4.0, 2.0])), 2.0 / 3.0)  # a4

    # test case based on tables I & IX from paper:
    # J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015
    def test_mdvf2(self):
        print('Starting MDVF test 2')
        model = MostDiscriminatingValueFunction(30, 4, True,
                                                best_obj=np.ones(4), worst_obj=np.full(4, fill_value=4.0))
        comparisons = [[np.array([2.0, 4.0, 4.0, 2.0]), np.array([4.0, 2.0, 2.0, 4.0])],  # a4 > a1
                       [np.array([4.0, 2.0, 2.0, 4.0]), np.array([3.0, 1.0, 3.0, 1.0])],  # a1 > a2
                       [np.array([2.0, 4.0, 4.0, 2.0]), np.array([3.0, 1.0, 3.0, 1.0])],  # a4 > a2
                       [np.array([2.0, 4.0, 4.0, 2.0]), np.array([1.0, 3.0, 1.0, 3.0])]]  # a4 > a3

        for pair in comparisons:
            model.update(pair)
        print(*model.interp_points, sep='\n')

        self.assertAlmostEqual(len(model.buffer), 2)  # check if model discarded 2 incompatible comparisons
        self.assertAlmostEqual(model.value_funtion(np.array([4.0, 2.0, 2.0, 4.0])), 0.0)  # a1
        self.assertAlmostEqual(model.value_funtion(np.array([3.0, 1.0, 3.0, 1.0])), 0.5)  # a2
        self.assertAlmostEqual(model.value_funtion(np.array([1.0, 3.0, 1.0, 3.0])), 0.5)  # a3
        self.assertAlmostEqual(model.value_funtion(np.array([2.0, 4.0, 4.0, 2.0])), 1.0)  # a4


class TestMaximalSumOfScoresValueFunction(TestCase):
    # test case based on tables I & VII from paper:
    # J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015
    def test_msvf(self):
        print('Starting MSVF test 1')
        model = MaximalSumOfScoresValueFunction(30, 4, True, const_eps=0.1,
                                                best_obj=np.ones(4), worst_obj=np.full(4, fill_value=4.0))
        comparisons = [[np.array([2.0, 4.0, 4.0, 2.0]), np.array([4.0, 2.0, 2.0, 4.0])],  # a4 > a1
                       [np.array([4.0, 2.0, 2.0, 4.0]), np.array([3.0, 1.0, 3.0, 1.0])],  # a1 > a2
                       [np.array([2.0, 4.0, 4.0, 2.0]), np.array([3.0, 1.0, 3.0, 1.0])]]  # a4 > a2

        for pair in comparisons:
            model.update(pair)
        print(*model.interp_points, sep='\n')

        self.assertEqual(len(model.buffer), 3)  # check if model buffered all 3 comparisons
        self.assertAlmostEqual(model.value_funtion(np.array([4.0, 2.0, 2.0, 4.0])), 0.45)   # a1
        self.assertAlmostEqual(model.value_funtion(np.array([3.0, 1.0, 3.0, 1.0])), 0.35)   # a2
        # the original paper assumes that all solutions are included in the "Maximal Sum of Scores" calculation
        # even those which are missing from the pairwise comparisons; here, only the solutions from the comparisons
        # are considered which slightly changes the utility of a3 from 1.0 to 0.825 (the remaining are the same)
        self.assertAlmostEqual(model.value_funtion(np.array([1.0, 3.0, 1.0, 3.0])), 0.825)  # a3
        self.assertAlmostEqual(model.value_funtion(np.array([2.0, 4.0, 4.0, 2.0])), 0.55)   # a4
