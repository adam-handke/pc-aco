import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD


class Model:
    def __init__(self, buffer, objectives, verbose):
        # buffer is implemented based on NEMO-0 approach
        # (J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015)
        # it is a list of pairs of vectors of objective values where the 1st is better than the 2nd according to the DM
        # when the buffer becomes too long or the preference model becomes incompatible, the oldest pair is discarded
        self.buffer_max_size = buffer
        self.buffer = []

        self.objectives = objectives
        self.verbose = verbose
        self.best_obj = None  # vector with lower bound for each objective
        self.worst_obj = None  # vector with upper bound for each objective

        # list of interpolation points dicts for every objective; for every objective (obj) should have the format of:
        # {'obj': [self.best_obj[obj], ..., self.worst_obj[obj]], 'util': [???, ..., 0.0]}
        self.interp_points = [{'obj': [], 'util': []} for _ in range(self.objectives)]

    def value_funtion(self, obj_val):
        # flexible additive monotonic value function model, separate piecewise-linear functions for every objective
        # stretched out on utility values calculated by linear programming for every individual from each pair
        # new values calculated as linear interpolation between sorted known values
        # assumes 'cost' type of objectives so 0.0 utility for 'worst_obj' and 1.0 utility for 'best_obj'
        # returns total utility for a vector of objective values of a single solution
        return np.sum([np.interp(obj_val[obj], self.interp_points[obj]['obj'], self.interp_points[obj]['util'])
                       for obj in range(self.objectives)])

    def update(self, compared_pair):
        # compared_pair = pair of vectors of objective values where the 1st is better than the 2nd according to the DM
        # different for every model
        pass

    def rank(self, objective_values):
        # double argsort = ranking
        # value function is negated because utilities are of type 'gain' and objectives are of type 'cost'
        ranked_solutions = np.argsort(np.argsort([-self.value_funtion(obj_val) for obj_val in objective_values]))
        return ranked_solutions


class MostDiscriminatingValueFunction(Model):
    def __str__(self):
        return 'MDVF'

    def update(self, compared_pair):
        if len(self.buffer) > self.buffer_max_size:
            raise ValueError(f'Too many pairs in the buffer ({len(self.buffer)})')
        elif len(self.buffer) == self.buffer_max_size:
            self.buffer.pop(0)
        self.buffer.append(compared_pair)

        while len(self.buffer) > 0:
            # LP definition based on PRVF problem equivalent to the MDVF problem as in the NEMO-0 approach
            # (J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015)
            lp = LpProblem(name=self.__str__(), sense=LpMaximize)
            u_better = [[LpVariable(f'ub{obj}_{pair}') for obj in range(self.objectives)]
                        for pair in range(len(self.buffer))]
            u_worse = [[LpVariable(f'uw{obj}_{pair}') for obj in range(self.objectives)]
                       for pair in range(len(self.buffer))]
            u_best = [LpVariable(f'ubest{obj}') for obj in range(self.objectives)]
            u_worst = [LpVariable(f'uworst{obj}') for obj in range(self.objectives)]
            epsilon = LpVariable('epsilon')

            # maximized variable
            lp += epsilon

            # monotonicity constraints
            for obj in range(self.objectives):
                for pair1 in range(len(self.buffer)):
                    for pair2 in range(len(self.buffer)):
                        for p1 in range(2):  # which solution in pair1
                            for p2 in range(2):  # which solution in pair2
                                if ((pair1 != pair2 or p1 != p2)
                                        and self.buffer[pair1][p1][obj] < self.buffer[pair2][p2][obj]):
                                    lp += ((u_better[pair1][obj] if p1 == 0 else u_worse[pair1][obj])
                                           >= (u_better[pair2][obj] if p2 == 0 else u_worse[pair2][obj]))
            for obj in range(self.objectives):
                for pair in range(len(self.buffer)):
                    lp += u_better[pair][obj] >= u_worst[obj]
                    lp += u_worse[pair][obj] >= u_worst[obj]
                    lp += u_best[obj] >= u_better[pair][obj]
                    lp += u_best[obj] >= u_worse[pair][obj]

            # pairwise preference constraints
            for pair in range(len(self.buffer)):
                lp += (sum([u_better[pair][obj] for obj in range(self.objectives)])
                       - sum([u_worse[pair][obj] for obj in range(self.objectives)])) >= epsilon

            # normalization constraint (lower bound)
            lp += sum([u_best[obj] for obj in range(self.objectives)]) == 1.0

            # normalization constraint (upper bound)
            for obj in range(self.objectives):
                lp += u_worst[obj] == 0.0

            lp.solve(PULP_CBC_CMD(msg=False))

            if epsilon.varValue <= 0:
                if self.verbose:
                    print(f'LP solved but the preferences are incompatible (epsilon={epsilon.varValue}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)')
                self.buffer.pop(0)
            else:
                if self.verbose:
                    print(f'LP solved and the preferences are compatible (epsilon={epsilon.varValue})')
                # translating LP results to interp_points
                self.interp_points = [{'obj': [self.best_obj[obj], self.worst_obj[obj]],
                                       'util': [u_best[obj].varValue, u_worst[obj].varValue]}
                                      for obj in range(self.objectives)]
                for obj in range(self.objectives):
                    for pair in range(len(self.buffer)):
                        for p in range(2):
                            self.interp_points[obj]['obj'].append(self.buffer[pair][p][obj])
                            self.interp_points[obj]['util'].append(u_better[pair][obj].varValue
                                                                   if p == 0 else u_worse[pair][obj].varValue)
                    # sort 'obj' ascending and 'util' descending
                    self.interp_points[obj]['obj'].sort()
                    self.interp_points[obj]['util'].sort(reverse=True)
                break


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
