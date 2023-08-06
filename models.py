import warnings
import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, PULP_CBC_CMD


class Model:
    def __init__(self, buffer_max_size, objectives, verbose, buffer=[], best_obj=None, worst_obj=None, const_eps=None):
        # buffer is implemented based on NEMO-0 approach
        # (J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015)
        # it is a list of pairs of vectors of objective values where the 1st is better than the 2nd according to the DM
        # when the buffer becomes too long or the preference model becomes incompatible, the oldest pair is discarded
        self.buffer_max_size = buffer_max_size
        self.buffer = buffer

        self.objectives = objectives
        self.verbose = verbose
        self.best_obj = best_obj  # vector with lower bound for each objective
        self.worst_obj = worst_obj  # vector with upper bound for each objective
        if const_eps is not None and const_eps <= 0.0:
            raise ValueError(f'wrong constant epsilon: {const_eps}; it should be a small positive float')
        self.const_eps = const_eps  # constant epsilon, only active when value given on initialization

        # list of interpolation points dicts for every objective; for every objective (obj) should have the format of:
        # {'obj': [self.best_obj[obj], ..., self.worst_obj[obj]], 'util': [???, ..., 0.0]}
        self.interp_points = [{'obj': [], 'util': []} for _ in range(self.objectives)]

    def value_funtion(self, obj_val):
        # flexible additive monotonic value function model, separate piecewise-linear functions for every objective
        # stretched out on utility values calculated by linear programming for every individual from each pair
        # new values calculated as linear interpolation between sorted known values
        # assumes 'cost' type of objectives so 0.0 utility for 'worst_obj' and up to 1.0 utility for 'best_obj'
        # returns total utility for a vector of objective values of a single solution
        return np.sum([np.interp(obj_val[obj], self.interp_points[obj]['obj'], self.interp_points[obj]['util'])
                       for obj in range(self.objectives)])

    def translate_interpolation_points(self, u_best, u_worst, u_better, u_worse):
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
        # print(*self.interp_points, sep='\n')

    def update(self, compared_pair):
        # compared_pair = pair of vectors of objective values where the 1st is better than the 2nd according to the DM
        # update procedure must be implemented differently for every model
        if len(self.buffer) > self.buffer_max_size:
            raise ValueError(f'Too many pairs in the buffer ({len(self.buffer)})')
        elif len(self.buffer) == self.buffer_max_size:
            self.buffer.pop(0)
        self.buffer.append(compared_pair)

    def rank(self, objective_values):
        # double argsort = ranking
        # value function is negated because utilities are of type 'gain' and objectives are of type 'cost'
        ranked_solutions = np.argsort(np.argsort([-self.value_funtion(obj_val) for obj_val in objective_values]))
        return ranked_solutions


class MostDiscriminatingValueFunction(Model):
    def __str__(self):
        return 'MDVF'

    def solve_linear_programming_problem(self):
        # LP definition based on the PRVF problem equivalent to the MDVF problem as in the NEMO-0 approach
        # (J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015)
        # defined as a separate method because it is used by MSCV and MSVF
        lp = LpProblem(name='PRVF/MDVF', sense=LpMaximize)
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
                            if pair1 != pair2 or p1 != p2:
                                if self.buffer[pair1][p1][obj] < self.buffer[pair2][p2][obj]:
                                    lp += ((u_better[pair1][obj] if p1 == 0 else u_worse[pair1][obj])
                                           >= (u_better[pair2][obj] if p2 == 0 else u_worse[pair2][obj]))
                                elif self.buffer[pair1][p1][obj] == self.buffer[pair2][p2][obj]:
                                    # making sure that the same objective value always has the same utility
                                    lp += ((u_better[pair1][obj] if p1 == 0 else u_worse[pair1][obj])
                                           == (u_better[pair2][obj] if p2 == 0 else u_worse[pair2][obj]))
        for obj in range(self.objectives):
            for pair in range(len(self.buffer)):
                if self.buffer[pair][0][obj] < self.worst_obj[obj]:
                    lp += u_better[pair][obj] >= u_worst[obj]
                elif self.buffer[pair][0][obj] == self.worst_obj[obj]:
                    lp += u_better[pair][obj] == u_worst[obj]

                if self.buffer[pair][1][obj] < self.worst_obj[obj]:
                    lp += u_worse[pair][obj] >= u_worst[obj]
                elif self.buffer[pair][1][obj] == self.worst_obj[obj]:
                    lp += u_worse[pair][obj] == u_worst[obj]

                if self.best_obj[obj] < self.buffer[pair][0][obj]:
                    lp += u_best[obj] >= u_better[pair][obj]
                elif self.best_obj[obj] == self.buffer[pair][0][obj]:
                    lp += u_best[obj] == u_better[pair][obj]

                if self.best_obj[obj] < self.buffer[pair][1][obj]:
                    lp += u_best[obj] >= u_worse[pair][obj]
                elif self.best_obj[obj] == self.buffer[pair][1][obj]:
                    lp += u_best[obj] == u_worse[pair][obj]
            lp += u_best[obj] >= u_worst[obj]

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
        return epsilon, u_best, u_worst, u_better, u_worse

    def update(self, compared_pair):
        super().update(compared_pair)
        while len(self.buffer) > 0:
            epsilon, u_best, u_worst, u_better, u_worse = self.solve_linear_programming_problem()
            if epsilon.varValue <= 0:
                if self.verbose:
                    print(f'LP solved but the preferences are incompatible (epsilon={epsilon.varValue}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')
            else:
                if self.verbose:
                    print(f'LP solved and the preferences are compatible (epsilon={epsilon.varValue})', flush=True)
                self.translate_interpolation_points(u_best, u_worst, u_better, u_worse)
                break


class MinimalSlopeChangeValueFunction(Model):
    def __str__(self):
        return 'MSCVF'

    def update(self, compared_pair):
        super().update(compared_pair)
        while len(self.buffer) > 0:
            # first LP - determine the epsilon using the PRVF/MDVF model
            if self.const_eps is None:
                eps_model = MostDiscriminatingValueFunction(self.buffer_max_size, self.objectives, False,
                                                            self.buffer, self.best_obj, self.worst_obj)
                epsilon, _, _, _, _ = eps_model.solve_linear_programming_problem()
                eps = epsilon.varValue
            else:
                eps = self.const_eps
            if eps <= 0:
                if self.verbose:
                    print(f'Eps-LP solved but the preferences are incompatible (epsilon={eps}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')
            else:
                if self.verbose and self.const_eps is None:
                    print(f'Eps-LP solved and the preferences are compatible (epsilon={eps}); '
                          f'now solving the main LP', flush=True)
                # second LP - solve the actual MSCVF problem using the known epsilon
                # just like in the paper (J. Branke et al.), epsilon is divided by 1000
                if self.const_eps is None:
                    eps /= 1000.0

                lp = LpProblem(name='MSVF', sense=LpMinimize)
                u_better = [[LpVariable(f'ub{obj}_{pair}') for obj in range(self.objectives)]
                            for pair in range(len(self.buffer))]
                u_worse = [[LpVariable(f'uw{obj}_{pair}') for obj in range(self.objectives)]
                           for pair in range(len(self.buffer))]
                u_best = [LpVariable(f'ubest{obj}') for obj in range(self.objectives)]
                u_worst = [LpVariable(f'uworst{obj}') for obj in range(self.objectives)]
                rho = LpVariable('rho')

                # minimized rho
                lp += rho

                # rho constraints - based on equation (5) from the paper
                for obj in range(self.objectives):
                    obj_util_list = [[self.best_obj[obj], u_best[obj]], [self.worst_obj[obj], u_worst[obj]]]
                    for pair in range(len(self.buffer)):
                        obj_util_list.append([self.buffer[pair][0][obj], u_better[pair][obj]])
                        obj_util_list.append([self.buffer[pair][1][obj], u_worse[pair][obj]])
                    obj_util_list.sort(key=lambda x: x[0])
                    for k in range(2, len(obj_util_list)):
                        if ((obj_util_list[k][0] - obj_util_list[k-1][0]) > 0
                                and (obj_util_list[k-1][0] - obj_util_list[k-2][0]) > 0):
                            lp += (((obj_util_list[k][1] - obj_util_list[k-1][1])
                                    / (obj_util_list[k][0] - obj_util_list[k-1][0]))
                                   - ((obj_util_list[k-1][1] - obj_util_list[k-2][1])
                                      / (obj_util_list[k-1][0] - obj_util_list[k-2][0]))) <= rho

                        if ((obj_util_list[k-1][0] - obj_util_list[k-2][0]) > 0
                                and (obj_util_list[k][0] - obj_util_list[k-1][0]) > 0):
                            lp += (((obj_util_list[k-1][1] - obj_util_list[k-2][1])
                                    / (obj_util_list[k-1][0] - obj_util_list[k-2][0]))
                                   - ((obj_util_list[k][1] - obj_util_list[k-1][1])
                                      / (obj_util_list[k][0] - obj_util_list[k-1][0]))) <= rho

                # monotonicity constraints
                for obj in range(self.objectives):
                    for pair1 in range(len(self.buffer)):
                        for pair2 in range(len(self.buffer)):
                            for p1 in range(2):  # which solution in pair1
                                for p2 in range(2):  # which solution in pair2
                                    if pair1 != pair2 or p1 != p2:
                                        if self.buffer[pair1][p1][obj] < self.buffer[pair2][p2][obj]:
                                            lp += ((u_better[pair1][obj] if p1 == 0 else u_worse[pair1][obj])
                                                   >= (u_better[pair2][obj] if p2 == 0 else u_worse[pair2][obj]))
                                        elif self.buffer[pair1][p1][obj] == self.buffer[pair2][p2][obj]:
                                            # making sure that the same objective value always has the same utility
                                            lp += ((u_better[pair1][obj] if p1 == 0 else u_worse[pair1][obj])
                                                   == (u_better[pair2][obj] if p2 == 0 else u_worse[pair2][obj]))
                for obj in range(self.objectives):
                    for pair in range(len(self.buffer)):
                        if self.buffer[pair][0][obj] < self.worst_obj[obj]:
                            lp += u_better[pair][obj] >= u_worst[obj]
                        elif self.buffer[pair][0][obj] == self.worst_obj[obj]:
                            lp += u_better[pair][obj] == u_worst[obj]

                        if self.buffer[pair][1][obj] < self.worst_obj[obj]:
                            lp += u_worse[pair][obj] >= u_worst[obj]
                        elif self.buffer[pair][1][obj] == self.worst_obj[obj]:
                            lp += u_worse[pair][obj] == u_worst[obj]

                        if self.best_obj[obj] < self.buffer[pair][0][obj]:
                            lp += u_best[obj] >= u_better[pair][obj]
                        elif self.best_obj[obj] == self.buffer[pair][0][obj]:
                            lp += u_best[obj] == u_better[pair][obj]

                        if self.best_obj[obj] < self.buffer[pair][1][obj]:
                            lp += u_best[obj] >= u_worse[pair][obj]
                        elif self.best_obj[obj] == self.buffer[pair][1][obj]:
                            lp += u_best[obj] == u_worse[pair][obj]
                    lp += u_best[obj] >= u_worst[obj]

                # pairwise preference constraints
                for pair in range(len(self.buffer)):
                    lp += (sum([u_better[pair][obj] for obj in range(self.objectives)])
                           - sum([u_worse[pair][obj] for obj in range(self.objectives)])) >= eps

                # normalization constraint (lower bound)
                lp += sum([u_best[obj] for obj in range(self.objectives)]) == 1.0

                # normalization constraint (upper bound)
                for obj in range(self.objectives):
                    lp += u_worst[obj] == 0.0

                lp.solve(PULP_CBC_CMD(msg=False))
                if self.verbose:
                    print(f'Main LP solved (rho={rho.varValue})', flush=True)
                self.translate_interpolation_points(u_best, u_worst, u_better, u_worse)
                break


class MaximalSumOfScoresValueFunction(Model):
    def __str__(self):
        return 'MSVF'

    def update(self, compared_pair):
        super().update(compared_pair)
        while len(self.buffer) > 0:
            # first LP - determine the epsilon using the PRVF/MDVF model
            if self.const_eps is None:
                eps_model = MostDiscriminatingValueFunction(self.buffer_max_size, self.objectives, False,
                                                            self.buffer, self.best_obj, self.worst_obj)
                epsilon, _, _, _, _ = eps_model.solve_linear_programming_problem()
                eps = epsilon.varValue
            else:
                eps = self.const_eps
            if eps <= 0:
                if self.verbose:
                    print(f'Eps-LP solved but the preferences are incompatible (epsilon={eps}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')
            else:
                if self.verbose and self.const_eps is None:
                    print(f'Eps-LP solved and the preferences are compatible (epsilon={eps}); '
                          f'now solving the main LP', flush=True)
                # second LP - solve the actual MSVF problem using the known epsilon
                # just like in the paper (J. Branke et al.), epsilon is divided by 1000
                if self.const_eps is None:
                    eps /= 1000.0

                lp = LpProblem(name='MSVF', sense=LpMaximize)
                u_better = [[LpVariable(f'ub{obj}_{pair}') for obj in range(self.objectives)]
                            for pair in range(len(self.buffer))]
                u_worse = [[LpVariable(f'uw{obj}_{pair}') for obj in range(self.objectives)]
                           for pair in range(len(self.buffer))]
                u_best = [LpVariable(f'ubest{obj}') for obj in range(self.objectives)]
                u_worst = [LpVariable(f'uworst{obj}') for obj in range(self.objectives)]

                # maximized sum of utilities (only includes the examples in pairs, not the whole population)
                lp += sum([sum(u_better[pair]) + sum(u_worse[pair]) for pair in range(len(self.buffer))])
                # additional components (a3 solution) of the sum needed to complete the example from the paper
                # more details in the TestMaximalSumOfScoresValueFunction.test_msvf() test case
                # + u_best[0] + 0.5 * (u_worse[0][1] + u_worst[1]) + u_best[2] + 0.5 * (u_better[0][3] + u_worst[3]))

                # monotonicity constraints
                for obj in range(self.objectives):
                    for pair1 in range(len(self.buffer)):
                        for pair2 in range(len(self.buffer)):
                            for p1 in range(2):  # which solution in pair1
                                for p2 in range(2):  # which solution in pair2
                                    if pair1 != pair2 or p1 != p2:
                                        if self.buffer[pair1][p1][obj] < self.buffer[pair2][p2][obj]:
                                            lp += ((u_better[pair1][obj] if p1 == 0 else u_worse[pair1][obj])
                                                   >= (u_better[pair2][obj] if p2 == 0 else u_worse[pair2][obj]))
                                        elif self.buffer[pair1][p1][obj] == self.buffer[pair2][p2][obj]:
                                            # making sure that the same objective value always has the same utility
                                            lp += ((u_better[pair1][obj] if p1 == 0 else u_worse[pair1][obj])
                                                   == (u_better[pair2][obj] if p2 == 0 else u_worse[pair2][obj]))
                for obj in range(self.objectives):
                    for pair in range(len(self.buffer)):
                        if self.buffer[pair][0][obj] < self.worst_obj[obj]:
                            lp += u_better[pair][obj] >= u_worst[obj]
                        elif self.buffer[pair][0][obj] == self.worst_obj[obj]:
                            lp += u_better[pair][obj] == u_worst[obj]

                        if self.buffer[pair][1][obj] < self.worst_obj[obj]:
                            lp += u_worse[pair][obj] >= u_worst[obj]
                        elif self.buffer[pair][1][obj] == self.worst_obj[obj]:
                            lp += u_worse[pair][obj] == u_worst[obj]

                        if self.best_obj[obj] < self.buffer[pair][0][obj]:
                            lp += u_best[obj] >= u_better[pair][obj]
                        elif self.best_obj[obj] == self.buffer[pair][0][obj]:
                            lp += u_best[obj] == u_better[pair][obj]

                        if self.best_obj[obj] < self.buffer[pair][1][obj]:
                            lp += u_best[obj] >= u_worse[pair][obj]
                        elif self.best_obj[obj] == self.buffer[pair][1][obj]:
                            lp += u_best[obj] == u_worse[pair][obj]
                    lp += u_best[obj] >= u_worst[obj]

                # pairwise preference constraints
                for pair in range(len(self.buffer)):
                    lp += (sum([u_better[pair][obj] for obj in range(self.objectives)])
                           - sum([u_worse[pair][obj] for obj in range(self.objectives)])) >= eps

                # normalization constraint (lower bound)
                lp += sum([u_best[obj] for obj in range(self.objectives)]) == 1.0

                # normalization constraint (upper bound)
                for obj in range(self.objectives):
                    lp += u_worst[obj] == 0.0

                lp.solve(PULP_CBC_CMD(msg=False))
                if self.verbose:
                    print(f'Main LP solved', flush=True)
                self.translate_interpolation_points(u_best, u_worst, u_better, u_worse)
                break


class RobustOrdinalRegression(Model):
    def __str__(self):
        return 'ROR'


class MonteCarlo(Model):
    def __str__(self):
        return 'MC'
