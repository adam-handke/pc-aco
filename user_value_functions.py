import numpy as np


class UserValueFunction:
    def __init__(self, weights, extreme_objective):
        if np.round(np.sum(weights), 7) == 1.0:
            self.weights = np.array(weights)
        else:
            raise ValueError(f'weights do not add up to 1.0: {weights} (sum={np.sum(weights)})')
        self.extreme_objective = extreme_objective

    def _function(self, objectives):
        raise NotImplementedError

    def calculate(self, objectives):
        if len(objectives) == len(self.weights):
            return self._function(objectives)
        else:
            raise ValueError(f'number of objectives ({len(objectives)}) is different than the '
                             f'number of weights ({len(self.weights)})')


class LinearUserValueFunction(UserValueFunction):
    def __str__(self):
        return 'Linear'

    def _function(self, objectives):
        return np.sum(self.weights * objectives)


class ChebycheffUserValueFunction(UserValueFunction):
    def __str__(self):
        return 'Chebycheff'

    def _function(self, objectives):
        return np.max(self.weights * objectives)
