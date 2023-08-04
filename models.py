import pulp
import numpy as np


class Model:
    def __init__(self, buffer, objectives, verbose):
        self.buffer_max_size = buffer
        self.buffer = []
        self.objectives = objectives
        self.verbose = verbose
        self.min_obj = None  # vector with lower bound for each objective
        self.max_obj = None  # vector with upper bound for each objective

    def value_funtion(self, objective_values):
        # TODO
        return np.mean(objective_values)

    def update(self, compared_pair):
        # compared_pair = pair of sets of objective values where the first is better than the second according to the DM
        # TODO
        pass

    def rank(self, objective_values):
        # double argsort = ranking
        ranked_solutions = np.argsort(np.argsort([self.value_funtion(obj) for obj in objective_values]))
        return ranked_solutions


class MostDiscriminatingValueFunction(Model):
    def __str__(self):
        return 'MDVF'


class MinimalSlopeChangeValueFunction(Model):
    def __str__(self):
        return 'MSCVF'


class MaximalSumOfScoresValueFunction(Model):
    def __str__(self):
        return 'MSVF'


class RobustOrdinalRegression(Model):
    def __str__(self):
        return 'ROR'


class MonteCarlo(Model):
    def __str__(self):
        return 'MC'
