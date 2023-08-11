import time
import warnings
import numpy as np
import multiprocessing
from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, PULP_CBC_CMD
from anyHR.constraint.Constraint import Constraints
from anyHR.hit_and_run.hit_and_run import HitAndRun, DirectionSampling, Shrinking, InitPoint


class Model:
    def __init__(self, buffer_max_size, objectives, verbose, buffer=None, best_obj=None, worst_obj=None, const_eps=None):
        # buffer is implemented based on NEMO-0 approach
        # (J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015)
        # it is a list of pairs of vectors of objective values where the 1st is better than the 2nd according to the DM
        # when the buffer becomes too long or the preference model becomes incompatible, the oldest pair is discarded
        self.buffer_max_size = buffer_max_size
        if buffer is None:
            self.buffer = list()
        else:
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
        # for Monte Carlo it's a list of lists of dicts (many value functions)
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
        interp_points = [{'obj': [self.best_obj[obj], self.worst_obj[obj]],
                          'util': [u_best[obj].varValue, u_worst[obj].varValue]} for obj in range(self.objectives)]
        for obj in range(self.objectives):
            for pair in range(len(self.buffer)):
                for p in range(2):
                    interp_points[obj]['obj'].append(self.buffer[pair][p][obj])
                    interp_points[obj]['util'].append(u_better[pair][obj].varValue
                                                      if p == 0 else u_worse[pair][obj].varValue)
            # sort 'obj' ascending and 'util' descending
            interp_points[obj]['obj'].sort()
            interp_points[obj]['util'].sort(reverse=True)
        # print(*interp_points, sep='\n')
        return interp_points

    def define_utilities(self):
        u_better = [[LpVariable(f'ub{obj}_{pair}') for obj in range(self.objectives)]
                    for pair in range(len(self.buffer))]
        u_worse = [[LpVariable(f'uw{obj}_{pair}') for obj in range(self.objectives)]
                   for pair in range(len(self.buffer))]
        u_best = [LpVariable(f'ubest{obj}') for obj in range(self.objectives)]
        u_worst = [LpVariable(f'uworst{obj}') for obj in range(self.objectives)]
        return u_better, u_worse, u_best, u_worst

    def add_pairwise_preference_constraints(self, lp, u_better, u_worse, epsilon):
        for pair in range(len(self.buffer)):
            lp += (sum([u_better[pair][obj] for obj in range(self.objectives)])
                   - sum([u_worse[pair][obj] for obj in range(self.objectives)])) >= epsilon
        return lp

    def add_monotonicity_constraints(self, lp, u_best, u_worst, u_better, u_worse):
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
        return lp

    def add_normalization_constraints(self, lp, u_best, u_worst, u_better, u_worse):
        # lower bound
        lp += sum([u_best[obj] for obj in range(self.objectives)]) == 1.0
        # upper bound
        for obj in range(self.objectives):
            lp += u_worst[obj] == 0.0
        # all >= 0.0
        for obj in range(self.objectives):
            for pair in range(len(self.buffer)):
                lp += u_better[pair][obj] >= 0.0
                lp += u_worse[pair][obj] >= 0.0
            lp += u_best[obj] >= 0.0
        # all <= 1.0
        for obj in range(self.objectives):
            for pair in range(len(self.buffer)):
                lp += u_better[pair][obj] <= 1.0
                lp += u_worse[pair][obj] <= 1.0
            lp += u_best[obj] <= 1.0
        return lp

    def update_upper_bound_on_objective(self, obj, new_upper_bound):
        self.worst_obj[obj] = new_upper_bound
        self.interp_points[obj]['obj'][-1] = new_upper_bound

    def update(self, compared_pair, init=False):
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
        # defined as a separate method because it is used by MSCVF, MSVF and ROR
        lp = LpProblem(name='PRVF/MDVF', sense=LpMaximize)
        u_better, u_worse, u_best, u_worst = self.define_utilities()
        epsilon = LpVariable('epsilon')

        # maximized variable
        lp += epsilon
        # pairwise preference constraints
        lp = self.add_pairwise_preference_constraints(lp, u_better, u_worse, epsilon)
        # monotonicity constraints
        lp = self.add_monotonicity_constraints(lp, u_best, u_worst, u_better, u_worse)
        # normalization constraints
        lp = self.add_normalization_constraints(lp, u_best, u_worst, u_better, u_worse)

        lp.solve(PULP_CBC_CMD(msg=False))
        return epsilon, u_best, u_worst, u_better, u_worse

    def update(self, compared_pair, init=False):
        super().update(compared_pair)
        while len(self.buffer) > 0:
            epsilon, u_best, u_worst, u_better, u_worse = self.solve_linear_programming_problem()
            if epsilon.varValue <= 0:
                if self.verbose:
                    print(f'LP solved but the preferences are incompatible (epsilon={np.round(epsilon.varValue, 5)}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')
            else:
                if self.verbose:
                    print(f'LP solved and the preferences are compatible (epsilon={np.round(epsilon.varValue, 5)})', flush=True)
                self.interp_points = self.translate_interpolation_points(u_best, u_worst, u_better, u_worse)
                break


class MinimalSlopeChangeValueFunction(Model):
    def __str__(self):
        return 'MSCVF'

    def update(self, compared_pair, init=False):
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
                    print(f'Eps-LP solved but the preferences are incompatible (epsilon={np.round(eps, 5)}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')
            else:
                if self.verbose and self.const_eps is None:
                    print(f'Eps-LP solved and the preferences are compatible (epsilon={np.round(eps, 5)}); '
                          f'now solving the main LP', flush=True)
                # second LP - solve the actual MSCVF problem using the known epsilon
                # just like in the paper (J. Branke et al., 2015), epsilon is divided by 1000
                if self.const_eps is None:
                    eps /= 1000.0

                lp = LpProblem(name='MSVF', sense=LpMinimize)
                u_better, u_worse, u_best, u_worst = self.define_utilities()
                rho = LpVariable('rho')

                # minimized rho
                lp += rho

                # rho constraints - based on equation (5) from the paper
                zero_threshold = 1e-10
                for obj in range(self.objectives):
                    obj_util_list = [[self.best_obj[obj], u_best[obj]], [self.worst_obj[obj], u_worst[obj]]]
                    for pair in range(len(self.buffer)):
                        obj_util_list.append([self.buffer[pair][0][obj], u_better[pair][obj]])
                        obj_util_list.append([self.buffer[pair][1][obj], u_worse[pair][obj]])
                    obj_util_list.sort(key=lambda x: x[0])
                    for k in range(2, len(obj_util_list)):
                        if ((obj_util_list[k][0] - obj_util_list[k-1][0]) > zero_threshold
                                and (obj_util_list[k-1][0] - obj_util_list[k-2][0]) > zero_threshold):
                            lp += (((obj_util_list[k][1] - obj_util_list[k-1][1])
                                    / (obj_util_list[k][0] - obj_util_list[k-1][0]))
                                   - ((obj_util_list[k-1][1] - obj_util_list[k-2][1])
                                      / (obj_util_list[k-1][0] - obj_util_list[k-2][0]))) <= rho

                        if ((obj_util_list[k-1][0] - obj_util_list[k-2][0]) > zero_threshold
                                and (obj_util_list[k][0] - obj_util_list[k-1][0]) > zero_threshold):
                            lp += (((obj_util_list[k-1][1] - obj_util_list[k-2][1])
                                    / (obj_util_list[k-1][0] - obj_util_list[k-2][0]))
                                   - ((obj_util_list[k][1] - obj_util_list[k-1][1])
                                      / (obj_util_list[k][0] - obj_util_list[k-1][0]))) <= rho
                # pairwise preference constraints
                lp = self.add_pairwise_preference_constraints(lp, u_better, u_worse, eps)
                # monotonicity constraints
                lp = self.add_monotonicity_constraints(lp, u_best, u_worst, u_better, u_worse)
                # normalization constraints
                lp = self.add_normalization_constraints(lp, u_best, u_worst, u_better, u_worse)

                lp.solve(PULP_CBC_CMD(msg=False))
                if self.verbose:
                    print(f'Main LP solved (rho={np.round(rho.varValue, 5)})', flush=True)
                self.interp_points = self.translate_interpolation_points(u_best, u_worst, u_better, u_worse)
                break


class MaximalSumOfScoresValueFunction(Model):
    def __str__(self):
        return 'MSVF'

    def update(self, compared_pair, init=False):
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
                    print(f'Eps-LP solved but the preferences are incompatible (epsilon={np.round(eps, 5)}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')
            else:
                if self.verbose and self.const_eps is None:
                    print(f'Eps-LP solved and the preferences are compatible (epsilon={np.round(eps, 5)}); '
                          f'now solving the main LP', flush=True)
                # second LP - solve the actual MSVF problem using the known epsilon
                # just like in the paper (J. Branke et al., 2015), epsilon is divided by 1000
                if self.const_eps is None:
                    eps /= 1000.0

                lp = LpProblem(name='MSVF', sense=LpMaximize)
                u_better, u_worse, u_best, u_worst = self.define_utilities()

                # maximized sum of utilities (only includes the examples in pairs, not the whole population)
                lp += sum([sum(u_better[pair]) + sum(u_worse[pair]) for pair in range(len(self.buffer))])
                # additional components (a3 solution) of the sum needed to complete the example from the paper
                # more details in the TestMaximalSumOfScoresValueFunction.test_msvf() test case
                # + u_best[0] + 0.5 * (u_worse[0][1] + u_worst[1]) + u_best[2] + 0.5 * (u_better[0][3] + u_worst[3]))

                # pairwise preference constraints
                lp = self.add_pairwise_preference_constraints(lp, u_better, u_worse, eps)
                # monotonicity constraints
                lp = self.add_monotonicity_constraints(lp, u_best, u_worst, u_better, u_worse)
                # normalization constraints
                lp = self.add_normalization_constraints(lp, u_best, u_worst, u_better, u_worse)

                lp.solve(PULP_CBC_CMD(msg=False))
                if self.verbose:
                    print(f'Main LP solved', flush=True)
                self.interp_points = self.translate_interpolation_points(u_best, u_worst, u_better, u_worse)
                break


class RobustOrdinalRegression(Model):
    def __str__(self):
        return 'ROR'

    def update(self, compared_pair, init=False):
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
                    print(f'Eps-LP solved but the preferences are incompatible (epsilon={np.round(eps, 5)}) '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')
            else:
                if self.verbose and self.const_eps is None:
                    print(f'Eps-LP solved and the preferences are compatible (epsilon={np.round(eps, 5)}); '
                          f'now solving the Necessary_preference-LP', flush=True)
                # second LP - solve the actual MSVF problem using the known epsilon
                # just like in the paper (J. Branke et al., 2015), epsilon is divided by 1000
                if self.const_eps is None:
                    eps /= 1000.0

                # second LP - checking if the new given pairwise comparison is necessary
                np_lp = LpProblem(name='necessary_preference', sense=LpMinimize)
                np_u_better, np_u_worse, np_u_best, np_u_worst = self.define_utilities()

                # minimize U(better) - U(worse) from the newly added pairwise comparison
                np_lp += sum(np_u_better[-1]) - sum(np_u_worse[-1])

                # pairwise preference constraints
                np_lp = self.add_pairwise_preference_constraints(np_lp, np_u_better, np_u_worse, eps)
                # monotonicity constraints
                np_lp = self.add_monotonicity_constraints(np_lp, np_u_best, np_u_worst, np_u_better, np_u_worse)
                # normalization constraints
                np_lp = self.add_normalization_constraints(np_lp, np_u_best, np_u_worst, np_u_better, np_u_worse)

                np_lp.solve(PULP_CBC_CMD(msg=False))
                # check if preference is necessary
                diff = (sum([np_u_better[-1][obj].varValue for obj in range(self.objectives)])
                        - sum([np_u_worse[-1][obj].varValue for obj in range(self.objectives)]))
                if diff >= 0:
                    if self.verbose:
                        print(f'Necessary_preference-LP solved (diff={np.round(diff, 5)})', flush=True)
                    # third LP - calculating the most representative value function
                    # joint minimization of delta from GRIP; as suggested in:
                    # J. Branke et al., Interactive Evolutionary Multiobjective Optimization Using ROR, 2009
                    lp = LpProblem(name='MRVF-ROR', sense=LpMinimize)
                    u_better, u_worse, u_best, u_worst = self.define_utilities()
                    delta = LpVariable('delta')
                    lp += delta

                    # pairwise necessary preference constraints (epsilon calculated earlier)
                    lp = self.add_pairwise_preference_constraints(lp, u_better, u_worse, eps)

                    # no necessary preference constraints (delta)
                    for pair1 in range(len(self.buffer)):
                        for pair2 in range(len(self.buffer)):
                            for p1 in range(2):  # which solution in pair1
                                for p2 in range(2):  # which solution in pair2
                                    if pair1 != pair2:
                                        lp += (((sum(u_better[pair1]) if p1 == 0 else sum(u_worse[pair1]))
                                               - (sum(u_better[pair2]) if p2 == 0 else sum(u_worse[pair2]))) <= delta)
                                        lp += (((sum(u_better[pair2]) if p1 == 0 else sum(u_worse[pair2]))
                                               - (sum(u_better[pair1]) if p2 == 0 else sum(u_worse[pair1]))) <= delta)

                    # monotonicity constraints
                    lp = self.add_monotonicity_constraints(lp, u_best, u_worst, u_better, u_worse)
                    # normalization constraints
                    lp = self.add_normalization_constraints(lp, u_best, u_worst, u_better, u_worse)

                    lp.solve(PULP_CBC_CMD(msg=False))
                    if self.verbose:
                        print(f'Main LP solved (delta={np.round(delta.varValue, 5)})', flush=True)
                    self.interp_points = self.translate_interpolation_points(u_best, u_worst, u_better, u_worse)
                    break
                else:
                    warnings.warn(f'Preference is not necessary - diff={np.round(diff, 5)} < 0; skipping model update')
                    self.buffer.pop(-1)


class MonteCarlo(Model):
    def __str__(self):
        return 'MC'

    def sample_worker(self, objectives, buffer, best_obj, worst_obj, queue):
        # worker defined as a subprocess in order to be able to terminate it if it runs for too long
        var_names = []
        var_names.extend([f'ubest{obj}' for obj in range(objectives)])
        var_names.extend([f'uworst{obj}' for obj in range(objectives)])
        for pair in range(len(buffer)):
            var_names.extend([f'ub{obj}_{pair}' for obj in range(objectives)])
        for pair in range(len(buffer)):
            var_names.extend([f'uw{obj}_{pair}' for obj in range(objectives)])

        bounds = [[-0.0000000001, 1.0000000001] for _ in range(len(var_names))]
        constr = Constraints(var_names)

        # pairwise preference constraints (no epsilon needed)
        for pair in range(len(buffer)):
            better_util = '+'.join([f'ub{obj}_{pair}' for obj in range(objectives)])
            worse_util = '+'.join([f'uw{obj}_{pair}' for obj in range(objectives)])
            constr.add_constraint(better_util + ' > ' + worse_util)

        # monotonicity constraints
        for obj in range(objectives):
            for pair1 in range(len(buffer)):
                for pair2 in range(len(buffer)):
                    for p1 in range(2):  # which solution in pair1
                        for p2 in range(2):  # which solution in pair2
                            if pair1 != pair2 or p1 != p2:
                                if buffer[pair1][p1][obj] < buffer[pair2][p2][obj]:
                                    constr.add_constraint((f'ub{obj}_{pair1}' if p1 == 0 else f'uw{obj}_{pair1}')
                                                          + ' >= '
                                                          + (f'ub{obj}_{pair2}' if p2 == 0 else f'uw{obj}_{pair2}'))
                                elif buffer[pair1][p1][obj] == buffer[pair2][p2][obj]:
                                    # making sure that the same objective value always has the same utility
                                    constr.add_constraint((f'ub{obj}_{pair1}' if p1 == 0 else f'uw{obj}_{pair1}')
                                                          + ' == '
                                                          + (f'ub{obj}_{pair2}' if p2 == 0 else f'uw{obj}_{pair2}'))
        for obj in range(objectives):
            for pair in range(len(buffer)):
                if buffer[pair][0][obj] < worst_obj[obj]:
                    constr.add_constraint(f'ub{obj}_{pair} >= uworst{obj}')
                elif buffer[pair][0][obj] == worst_obj[obj]:
                    constr.add_constraint(f'ub{obj}_{pair} == uworst{obj}')

                if buffer[pair][1][obj] < worst_obj[obj]:
                    constr.add_constraint(f'uw{obj}_{pair} >= uworst{obj}')
                elif buffer[pair][1][obj] == worst_obj[obj]:
                    constr.add_constraint(f'uw{obj}_{pair} == uworst{obj}')

                if best_obj[obj] < buffer[pair][0][obj]:
                    constr.add_constraint(f'ubest{obj} >= ub{obj}_{pair}')
                elif best_obj[obj] == buffer[pair][0][obj]:
                    constr.add_constraint(f'ubest{obj} == ub{obj}_{pair}')

                if best_obj[obj] < buffer[pair][1][obj]:
                    constr.add_constraint(f'ubest{obj} >= uw{obj}_{pair}')
                elif best_obj[obj] == buffer[pair][1][obj]:
                    constr.add_constraint(f'ubest{obj} == uw{obj}_{pair}')
            constr.add_constraint(f'ubest{obj} >= uworst{obj}')

        # normalization constraints (lower bound)
        constr.add_constraint('+'.join([f'ubest{obj}' for obj in range(objectives)]) + ' == 1.0')
        # normalization constraints (upper bound)
        for obj in range(objectives):
            constr.add_constraint(f'uworst{obj} == 0.0')

        har = HitAndRun(constraint=constr, bounding_box=bounds, direction_sampling=DirectionSampling.CDHR,
                        shrinking=Shrinking.SHRINKING, init_point=InitPoint.SMT)
        sample, rejections = har.next_sample()
        queue.put(sample)

    def update(self, compared_pair, init=False):
        super().update(compared_pair)
        while len(self.buffer) > 0:
            # custom three-step Monte Carlo process done using anyHR package
            # 1) define variables (utility values) and their constraints (preference, monotonicity, normalization)
            # 2) sample compatible value functions using hit-and-run method
            # 3) save all sampled value functions and later average the utilities over all of them
            samples = []
            try:
                num_samples = 10
                time_limit = 10  # 10 seconds time limit
                queue = multiprocessing.SimpleQueue()
                start = time.perf_counter()
                # stop sampling conditions:
                # 1) 10 samples achieved
                # 2) 10s total time limit of sampling exceeded (but this isn't first sampling during initialization)
                # 3) 10s time limit of a single sampling exceeded causes that sampling to be killed
                while len(samples) < num_samples and ((init and len(samples) == 0)
                                                      or time.perf_counter() - start < time_limit):
                    p = multiprocessing.Process(target=self.sample_worker, args=(self.objectives, self.buffer,
                                                                                 self.best_obj, self.worst_obj, queue))
                    p.daemon = True
                    p.start()
                    p.join(timeout=time_limit)
                    if p.is_alive():
                        warnings.warn('Sampling timed out!')
                        p.kill()
                        p.join()
                    else:
                        sample = queue.get()
                        samples.append(sample)
                if len(samples) == 0:
                    raise ValueError('Sampling timed out before returning any samples!')
                else:
                    self.interp_points = [self.translate_interpolation_points_from_flat(flat_input=s) for s in samples]
                    if self.verbose:
                        print(f'Sampled and saved {len(samples)} value functions', flush=True)
                break
            except Exception as e:
                warnings.warn(str(e))
                if self.verbose:
                    print(f'Unable to sample compatible value functions '
                          f'- discarding the oldest pair ({len(self.buffer) - 1} will remain)', flush=True)
                self.buffer.pop(0)
                if len(self.buffer) == 0:
                    warnings.warn('Unable to perform model update - no pairs left in the buffer!')

    def translate_interpolation_points_from_flat(self, flat_input):
        # translating sampling results to interp_points
        interp_points = [{'obj': [self.best_obj[obj], self.worst_obj[obj]],
                          'util': [flat_input[0 + obj], flat_input[self.objectives + obj]]}
                         for obj in range(self.objectives)]
        for obj in range(self.objectives):
            for pair in range(len(self.buffer)):
                for p in range(2):
                    interp_points[obj]['obj'].append(self.buffer[pair][p][obj])
                    interp_points[obj]['util'].append(flat_input[(2 + pair) * self.objectives + obj]
                                                      if p == 0 else flat_input[(2 + len(self.buffer) + pair)
                                                                                * self.objectives + obj])
            # sort 'obj' ascending and 'util' descending
            interp_points[obj]['obj'].sort()
            interp_points[obj]['util'].sort(reverse=True)
        # print(*interp_points, sep='\n')
        return interp_points

    def value_funtion(self, obj_val):
        # average over all saved value functions
        return np.mean([np.sum([np.interp(obj_val[obj], vf[obj]['obj'], vf[obj]['util'])
                                for obj in range(self.objectives)]) for vf in self.interp_points])

    def update_upper_bound_on_objective(self, obj, new_upper_bound):
        self.worst_obj[obj] = new_upper_bound
        for vf in range(len(self.interp_points)):
            self.interp_points[vf][obj]['obj'][-1] = new_upper_bound
