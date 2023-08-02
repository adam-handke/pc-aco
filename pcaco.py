import argparse
import time
import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
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
            self.user_value_function = LinearUserValueFunction(weights, extreme_objective)
        elif user_value_function == 'chebycheff':
            self.user_value_function = ChebycheffUserValueFunction(weights, extreme_objective)
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
        self.nds = NonDominatedSorting()
        self.aco_means = None  # 2d matrix (self.ants, self.variables)
        self.aco_weights = None  # 1d vector (self.ants), don't confuse with user value function weights
        self.aco_weight_sum = None
        self.aco_probabilities = None
        # self.stds = None  # 2d matrix (self.ants, self.variables)
        # no need to store stds as they can be calculated at construction

        # pre-calculation of all possible weight values as they only depend on ranks
        # based on equation (7) from K. Socha & M. Dorigo, Ant colony optimization for continuous domains, 2008
        self.weight_constant1 = 1.0 / (self.q * self.ants * np.sqrt(2.0 * np.pi))
        self.weight_constant2 = - 1.0 / (2.0 * self.q**2 * self.ants**2)
        self.rank_weight_dict = {rank: self.weight_constant1 * np.exp(rank**2 * self.weight_constant2)
                                 for rank in range(self.ants)}

    def update_ant_colony(self, population, objective_values):
        # mean update
        # based on equation (6) from K. Socha & M. Dorigo, Ant colony optimization for continuous domains, 2008
        self.aco_means = population

        # weight update
        if self.with_nondominance_ranking:
            # TODO
            ranked_population = self.model.rank(objective_values)
        else:
            ranked_population = self.model.rank(objective_values)
        self.aco_weights = np.array([self.rank_weight_dict[rank] for rank in ranked_population])
        self.aco_weight_sum = np.sum(self.aco_weights)
        # based on equation (8) from K. Socha & M. Dorigo, Ant colony optimization for continuous domains, 2008
        self.aco_probabilities = self.aco_weights / self.aco_weight_sum

        # standard deviation is calculated at construction to save time (not selected stds can be ignored)
        # self.aco_stds = np.zeros((self.ants, self.variables))
        # for k in range(self.ants):
        #     for n in range(self.variables):
        #         self.aco_stds[k, n] = self.xi * np.sum(np.abs(population[:, n] - population[k, n])) / (self.ants - 1)

    def update_preference_model(self, objective_values):
        # randomly select 2 non-dominated solutions and ask DM for comparison
        non_dominated_solutions = self.nds.do(objective_values, only_non_dominated_front=True)
        selected_index1, selected_index2 = self.rng.choice(non_dominated_solutions, size=2, replace=False)
        obj1, obj2 = objective_values[selected_index1], objective_values[selected_index2]
        val1 = self.user_value_function.calculate(obj1)
        val2 = self.user_value_function.calculate(obj2)
        if val1 >= val2:
            compared_pair = [obj1, obj2]
        else:
            compared_pair = [obj2, obj1]
        self.model.update(compared_pair)

    def reflected_normal_distribution(self, loc, scale, min_val, max_val):
        # safely get a number from normal distribution
        # in case the value is out of the allowed bounds, reflect the value on the lower/upper bound
        # in case the value is still out of bounds, trim to the lower/upper bound
        base_val = self.rng.normal(loc, scale)
        if base_val < min_val:
            return min(max(min_val - base_val, min_val), max_val)
        elif base_val > max_val:
            return max(min(2.0 * max_val - base_val, max_val), min_val)
        else:
            return base_val

    def construct_solution(self, population):
        # construction loop for each variable separately
        solution = np.zeros(self.variables)
        for n in range(self.variables):
            # step 1: randomly select the gaussian function
            k = self.rng.choice(range(self.ants), size=1, p=self.aco_probabilities)

            # step 2: randomly sample the value from a parametrized normal distribution
            # calculate std for selected gaussian function
            # based on equation (9) from K. Socha & M. Dorigo, Ant colony optimization for continuous domains, 2008
            std = self.xi * np.sum(np.abs(population[:, n] - population[k, n])) / (self.ants - 1)
            solution[n] = self.reflected_normal_distribution(loc=self.aco_means[k, n], scale=std,
                                                             min_val=self.problem.xl[n],
                                                             max_val=self.problem.xu[n]).item()
        return np.array(solution)

    def save(self, objective_values):
        convergence_indicators = [self.user_value_function.calculate(obj) for obj in objective_values]
        results_df = pd.DataFrame({
            'start_time': self.start_time,
            'seed': self.seed,
            'generations': self.generations,
            'ants': self.ants,
            'q': self.q,
            'xi': self.xi,
            'interval': self.interval,
            'buffer': self.buffer,
            'problem': self.problem.__class__.__name__,
            'variables': self.variables,
            'objectives': self.objectives,
            'user_value_function': str(self.user_value_function),
            'extreme_objective': self.user_value_function.extreme_objective,
            'model': str(self.model),
            'with_nondominance_ranking': self.with_nondominance_ranking,
            'min_convergence': np.min(convergence_indicators),
            'avg_convergence': np.mean(convergence_indicators),
            'duration': self.duration
        }, index=[0])
        results_df.to_csv('results/results.csv', mode='a', sep=';', index=False, header=False)

    def plot(self, objective_values):
        if self.objectives == 2:
            plt.figure(figsize=(10, 10))
            plt.scatter(objective_values[:, 0], objective_values[:, 1])
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.savefig(f'results/plot_{self.start_time}.png')
            plt.close()
        else:
            print('Plotting error - only instances with 2 objectives can be plotted.', flush=True)

    def solve(self):
        start = time.perf_counter()

        # random initialization of the 0th population
        population = self.rng.uniform(low=np.tile(self.problem.xl, (self.ants, 1)),
                                      high=np.tile(self.problem.xu, (self.ants, 1)),
                                      size=(self.ants, self.variables))
        objective_values = self.problem.evaluate(population)
        self.update_preference_model(objective_values)
        self.update_ant_colony(population, objective_values)
        if self.verbose:
            convergence_indicators = [self.user_value_function.calculate(obj) for obj in objective_values]
            print(f'Finished generation 0 after {np.round(time.perf_counter() - start, 3)}s from start '
                  f'(min={np.round(np.min(convergence_indicators), 3)}; '
                  f'avg={np.round(np.mean(convergence_indicators), 3)})', flush=True)

        # main ACO loop
        for g in range(1, self.generations+1):
            new_population = np.array([self.construct_solution(population) for _ in range(self.ants)])
            new_objective_values = self.problem.evaluate(new_population)
            if g % self.interval == 0:
                self.update_preference_model(new_objective_values)
            self.update_ant_colony(new_population, new_objective_values)

            population = new_population
            objective_values = new_objective_values

            if self.verbose:
                convergence_indicators = [self.user_value_function.calculate(obj) for obj in objective_values]
                print(f'Finished generation {g} after {np.round(time.perf_counter() - start, 3)}s from start '
                      f'(min={np.round(np.min(convergence_indicators), 3)}; '
                      f'avg={np.round(np.mean(convergence_indicators), 3)})', flush=True)

        stop = time.perf_counter()
        self.duration = stop - start
        if self.save_csv:
            self.save(objective_values)
        if self.draw_plot:
            self.plot(objective_values)
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
