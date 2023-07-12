import argparse
import time
import numpy as np
from pymoo.problems import get_problem
import matplotlib.pyplot as plt

from user_value_functions import LinearUserValueFunction, ChebycheffUserValueFunction
from models import MostDiscriminatingValueFunction, MinimalSlopeChangeValueFunction, MaximalSumOfScoresValueFunction, \
    RobustOrdinalRegression, MonteCarlo


class PairwiseComparisonsBasedAntColonyOptimization:
    def __init__(self, generations=100, ants=100, q=0.1, xi=0.5, interval=10, buffer=50, problem='zdt1', variables=None,
                 objectives=None, user_value_function='linear', extreme_objective=False, model='mdvf',
                 with_nondominance_ranking=True, seed=42, save_csv=True, draw_plot=True, verbose=False):
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # handling given parameters
        if isinstance(generations, int) and generations > 0:
            self.generations = generations
        else:
            raise ValueError(f'wrong number of generations: {generations}')

        if isinstance(ants, int) and ants > 0:
            self.ants = ants
        else:
            raise ValueError(f'wrong number of ants: {ants}')

        if isinstance(q, float) and 0.0 < q < 1.0:
            self.q = q
        else:
            raise ValueError(f'wrong diversification parameter q: {q}')

        if isinstance(xi, float) and 0.0 < xi < 1.0:
            self.xi = xi
        else:
            raise ValueError(f'wrong convergence parameter xi: {xi}')

        if isinstance(interval, int) and 0 < interval < generations:
            self.interval = interval
        else:
            raise ValueError(f'wrong length of interval: {interval}')

        if isinstance(buffer, int) and buffer > 0:
            self.buffer = buffer
        else:
            raise ValueError(f'wrong buffer size: {buffer}')

        if problem in ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7',
                       'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9']:
            if isinstance(variables, int) and variables > 0:
                self.variables = variables
            else:
                raise ValueError(f'wrong number of variables: {variables}')

            if isinstance(objectives, int) and 1 < objectives < 5:
                self.objectives = objectives
            else:
                raise ValueError(f'wrong number of objectives: {objectives}')

            self.problem = get_problem(problem, n_var=self.variables, n_obj=self.objectives)
        elif problem in ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt5', 'zdt6']:
            self.problem = get_problem(problem)
            self.variables = self.problem.n_var
            self.objectives = self.problem.n_obj
        else:
            raise ValueError(f'unsupported problem type: {problem}')

        # weights for cases with 2 or 4 objectives taken from 2014 paper by Branke, Greco, Slowinski, Zielniewicz
        weights_dict = {'regular': {2: [0.6, 0.4], 3: [0.4, 0.25, 0.35], 4: [0.3, 0.15, 0.2, 0.35]},
                        'extreme': {2: [0.85, 0.15], 3: [0.7, 0.1, 0.2], 4: [0.65, 0.1, 0.15, 0.1]}}
        if isinstance(extreme_objective, bool) and extreme_objective:
            weights = weights_dict['extreme'][self.objectives]
        else:
            weights = weights_dict['regular'][self.objectives]

        if user_value_function == 'linear':
            self.user_value_function = LinearUserValueFunction(weights)
        elif user_value_function == 'chebycheff':
            self.user_value_function = ChebycheffUserValueFunction(weights)
        else:
            raise ValueError(f'unknown user value function: {user_value_function}')

        if model == 'mdvf':
            self.model = MostDiscriminatingValueFunction()
        elif model == 'mscvf':
            self.model = MinimalSlopeChangeValueFunction()
        elif model == 'msvf':
            self.model = MaximalSumOfScoresValueFunction()
        elif model == 'ror':
            self.model = RobustOrdinalRegression()
        elif model == 'mc':
            self.model = MonteCarlo()
        else:
            raise ValueError(f'unknown model: {model}')

        if isinstance(with_nondominance_ranking, bool):
            self.with_nondominance_ranking = with_nondominance_ranking

        if isinstance(seed, int):
            self.seed = seed

        if isinstance(save_csv, bool):
            self.save_csv = save_csv

        if isinstance(draw_plot, bool):
            self.draw_plot = draw_plot

        if isinstance(verbose, bool):
            self.verbose = verbose

        if self.verbose:
            print(f'PC-ACO initialized successfully at {start_time} with parameters:', flush=True)
            for key, value in self.__dict__.items():
                if isinstance(value, int) or isinstance(value, str) or isinstance(value, bool) \
                        or isinstance(value, float):
                    print(f'\t{key}:'.ljust(25) + f'\t{value}', flush=True)
                else:
                    print(f'\t{key}:'.ljust(25) + f'\t{value.__class__.__name__}', flush=True)
        self.start_time = start_time
        self.duration = 0.0

        # initializing ant colony
        self.rng = np.random.default_rng(seed)
        self.means = self.rng.uniform(low=np.tile(self.problem.xl, (self.ants, 1)),
                                      high=np.tile(self.problem.xu, (self.ants, 1)),
                                      size=(self.ants, self.variables))
        self.stds = np.ones((self.ants, self.variables)) / 10.0
        self.weights = np.ones(self.ants) / self.ants

    def construct_solution(self):
        # construction loop
        for n in range(self.variables):
            # TODO
            pass
        return []

    def sort_solutions(self, solutions):
        # TODO
        sorted_solutions = self.model.sort(solutions)
        return sorted_solutions

    def update_ant_colony(self):
        # TODO
        pass

    def save(self):
        # TODO
        pass

    def plot(self):
        plt.figure(figsize=(10, 10))
        # TODO
        plt.close()
        plt.savefig(f'results/plot_{self.start_time}.png')

    def solve(self):
        start = time.perf_counter()
        # main ACO loop
        for g in range(1, self.generations+1):
            population = [self.construct_solution() for k in range(self.ants)]
            sorted_population = self.sort_solutions(population)
            self.update_ant_colony()
            # TODO

        stop = time.perf_counter()
        self.duration = stop - start
        if self.save_csv:
            self.save()
            # TODO
        if self.draw_plot:
            self.plot()
            # TODO
        if self.verbose:
            print(f'PC-ACO completed optimization successfully in {np.round(self.duration, 3)}s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generations', type=int, default=100, help='number of ACO generations')
    parser.add_argument('-a', '--ants', type=int, default=100, help='number of ants and population size')
    parser.add_argument('-q', type=float, default=0.1, help='diversification parameter of ACO')
    parser.add_argument('-xi', type=float, default=0.5, help='convergence parameter of ACO')
    parser.add_argument('-i', '--interval', type=int, default=10,
                        help='intervals between asking the user for a pairwise comparison; interval<generations')
    parser.add_argument('-b', '--buffer', type=int, default=50,
                        help='max number of stored pairwise comparisons')
    parser.add_argument('-p', '--problem', default='zdt1',
                        choices=['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7',
                                 'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9',
                                 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt5', 'zdt6'],
                        help='type of MOO benchamark problem to solve (from pymoo library)')
    parser.add_argument('-v', '--variables', type=int, default=10,
                        help='number of variables for DTLZ and WFG problems (not applicable for ZDT)')
    parser.add_argument('-o', '--objectives', type=int, default=2,
                        help='number of objectives (2, 3 or 4) for DTLZ and WFG problems (not applicable for ZDT)')
    parser.add_argument('-u', '--user-value-function', choices=['linear', 'chebycheff'], default='linear',
                        help='type of the user`s (decision maker`s) value function for comparisons')
    parser.add_argument('-e', '--extreme-objective', action='store_true',
                        help='the user considers one of the objectives as much more important than all others')
    parser.add_argument('-m', '--model', choices=['mdvf', 'mscvf', 'msvf', 'ror', 'mc'], default='mdvf',
                        help='type of the value function approach for the preference model')
    parser.add_argument('-r', '--with-nondominance-ranking', action='store_true',
                        help='whether to use the nondominance ranking during solution sorting')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='random number generator seed')
    parser.add_argument('-c', '--save-csv', action='store_true',
                        help='save results to csv')
    parser.add_argument('-d', '--draw-plot', action='store_true',
                        help='draw plot of the results')
    parser.add_argument('--verbose', action='store_true',
                        help='print messages to terminal')
    args = parser.parse_args()

    pcaco = PairwiseComparisonsBasedAntColonyOptimization(generations=args.generations,
                                                          ants=args.ants,
                                                          q=args.q,
                                                          xi=args.xi,
                                                          interval=args.interval,
                                                          buffer=args.buffer,
                                                          problem=args.problem,
                                                          variables=args.variables,
                                                          objectives=args.objectives,
                                                          user_value_function=args.user_value_function,
                                                          extreme_objective=args.extreme_objective,
                                                          model=args.model,
                                                          with_nondominance_ranking=args.with_nondominance_ranking,
                                                          seed=args.seed,
                                                          save_csv=args.save_csv,
                                                          draw_plot=args.draw_plot,
                                                          verbose=args.verbose)
    pcaco.solve()
