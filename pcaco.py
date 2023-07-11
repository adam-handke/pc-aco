import argparse
import time
from pymoo.problems import get_problem

from user_value_functions import LinearUserValueFunction, ChebycheffUserValueFunction
from models import MostDiscriminatingValueFunction, MinimalSlopeChangeValueFunction, MaximalSumOfScoresValueFunction, \
    RobustOrdinalRegression, MonteCarlo


class PairwiseComparisonsBasedAntColonyOptimization:
    def __init__(self, generations=100, interval=10, buffer=50, problem='zdt1', variables=None, objectives=None,
                 user_value_function='linear', extreme_objective=False, model='mdvf', with_nondominance_ranking=True,
                 save_csv=True, draw_plot=True, verbose=False):
        if isinstance(generations, int) and generations > 0:
            self.generations = generations
        else:
            raise ValueError(f'wrong number of generations: {generations}')

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

        if isinstance(save_csv, bool):
            self.save_csv = save_csv

        if isinstance(draw_plot, bool):
            self.draw_plot = draw_plot

        if isinstance(verbose, bool):
            self.verbose = verbose

        self.start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        if self.verbose:
            print(f'PC-ACO initialized successfully at {self.start_time} with parameters:', flush=True)
            for key, value in self.__dict__.items():
                if isinstance(value, int) or isinstance(value, str) or isinstance(value, bool):
                    print(f'\t{key}:'.ljust(25) + f'\t{value}', flush=True)
                else:
                    print(f'\t{key}:'.ljust(25) + f'\t{value.__class__.__name__}', flush=True)

    def solve(self):
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generations', type=int, default=100, help='number of generations')
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
    parser.add_argument('-s', '--save-csv', action='store_true',
                        help='save results to csv')
    parser.add_argument('-d', '--draw-plot', action='store_true',
                        help='draw plot of the results')
    parser.add_argument('--verbose', action='store_true',
                        help='print messages to terminal')
    args = parser.parse_args()

    pcaco = PairwiseComparisonsBasedAntColonyOptimization(generations=args.generations,
                                                          interval=args.interval,
                                                          buffer=args.buffer,
                                                          problem=args.problem,
                                                          variables=args.variables,
                                                          objectives=args.objectives,
                                                          user_value_function=args.user_value_function,
                                                          extreme_objective=args.extreme_objective,
                                                          model=args.model,
                                                          with_nondominance_ranking=args.with_nondominance_ranking,
                                                          save_csv=args.save_csv,
                                                          draw_plot=args.draw_plot,
                                                          verbose=args.verbose)
    pcaco.solve()
