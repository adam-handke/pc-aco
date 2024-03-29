import argparse
import time
import os
import random
import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
from matplotlib import font_manager

from user_value_functions import LinearUserValueFunction, ChebycheffUserValueFunction
from models import MostDiscriminatingValueFunction, MinimalSlopeChangeValueFunction, MaximalSumOfScoresValueFunction, \
    MostRepresentativeValueFunction, MonteCarlo


class PairwiseComparisonsBasedAntColonyOptimization:
    def __init__(self, generations=100, ants=30, q=0.1, xi=0.5, interval=10, buffer=30, problem='wfg4', variables=10,
                 objectives=2, user_value_function='linear', extreme_objective=False, model='mdvf',
                 without_nondominance_ranking=False, max_no_improvement=20, seed=42, dont_save_csv=False,
                 results_file='results/results.csv', draw_plot=False, plotting_checkpoints=None, plotting_ticks=None,
                 draw_value_function=False, pdf_plots=False, plot_file=None, vf_plot_file=None, verbose=False):
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

            if isinstance(objectives, int) and 1 < objectives < 6:
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

        # weights for cases with 2 or 4 objectives taken from 2014 paper:
        # J. Branke et al., Learning Value Functions in Interactive Evolutionary Multiobjective Optimization, 2015
        weights_dict = {'regular': {2: [0.6, 0.4], 3: [0.4, 0.25, 0.35], 4: [0.3, 0.15, 0.2, 0.35], 5: [0.25, 0.15, 0.2, 0.3, 0.1]},
                        'extreme': {2: [0.85, 0.15], 3: [0.7, 0.1, 0.2], 4: [0.65, 0.1, 0.15, 0.1], 5: [0.6, 0.1, 0.15, 0.1, 0.05]}}
        if isinstance(extreme_objective, bool) and extreme_objective:
            weights = weights_dict['extreme'][self.objectives]
            self.extreme_objective = extreme_objective
        else:
            weights = weights_dict['regular'][self.objectives]
            self.extreme_objective = extreme_objective

        if user_value_function == 'linear':
            self.user_value_function = LinearUserValueFunction(weights)
        elif user_value_function == 'chebycheff':
            self.user_value_function = ChebycheffUserValueFunction(weights)
        else:
            raise ValueError(f'unknown user value function: {user_value_function}')

        if isinstance(verbose, bool):
            self.verbose = verbose

        if model == 'mdvf':
            self.model = MostDiscriminatingValueFunction(self.buffer, self.objectives, self.verbose)
        elif model == 'mscvf':
            self.model = MinimalSlopeChangeValueFunction(self.buffer, self.objectives, self.verbose)
        elif model == 'msvf':
            self.model = MaximalSumOfScoresValueFunction(self.buffer, self.objectives, self.verbose)
        elif model == 'mrvf':
            self.model = MostRepresentativeValueFunction(self.buffer, self.objectives, self.verbose)
        elif model == 'mc':
            self.model = MonteCarlo(self.buffer, self.objectives, self.verbose)
        else:
            raise ValueError(f'unknown model: {model}')

        if isinstance(without_nondominance_ranking, bool):
            self.without_nondominance_ranking = without_nondominance_ranking
        else:
            raise ValueError(f'wrong `without_nondominance_ranking` parameter: {without_nondominance_ranking}')

        if isinstance(max_no_improvement, int):
            self.max_no_improvement = max_no_improvement
        else:
            raise ValueError(f'wrong `max_no_improvement` parameter: {max_no_improvement}')

        if isinstance(seed, int):
            self.seed = seed
        else:
            raise ValueError(f'wrong `seed` parameter: {seed}')

        if isinstance(dont_save_csv, bool):
            self.dont_save_csv = dont_save_csv
        else:
            raise ValueError(f'wrong `dont_save_csv` parameter: {dont_save_csv}')

        if isinstance(results_file, str):
            self.results_file = results_file
        else:
            raise ValueError(f'wrong `results_file` parameter: {results_file}')

        if isinstance(draw_plot, bool):
            self.draw_plot = draw_plot
            self.plotting_ticks = plotting_ticks
        else:
            raise ValueError(f'wrong `draw_plot` parameter: {draw_plot}')

        if plotting_checkpoints is not None and len(plotting_checkpoints) == 4:
            self.plotting_checkpoints = plotting_checkpoints
        else:
            # set plotting checkpoint at 10%, 1/3, 2/3 and 100% of the number of generations
            self.plotting_checkpoints = [int(np.round(frac * self.generations, 0))
                                         for frac in [0.1, 1.0 / 3.0, 2.0 / 3.0, 1.0]]
        self.color_dict = {gen: color for gen, color in zip(self.plotting_checkpoints,
                                                            ['tab:green', 'tab:blue', 'tab:purple', 'tab:red'])}
        self.marker_dict = {gen: shape for gen, shape in zip(self.plotting_checkpoints, ['^', 's', 'p', 'o'])}

        if isinstance(draw_value_function, bool):
            self.draw_value_function = draw_value_function
        else:
            raise ValueError(f'wrong `draw_value_function` parameter: {draw_value_function}')

        if isinstance(pdf_plots, bool):
            self.pdf_plots = pdf_plots
        else:
            raise ValueError(f'wrong `pdf_plots` parameter: {pdf_plots}')

        if self.draw_plot or self.draw_value_function:
            # turn on latex font if available
            # can be downloaded from: https://www.fontsquirrel.com/fonts/computer-modern
            if 'cmr10' in font_manager.get_font_names():
                plt.rcParams.update({'font.family': 'serif',
                                     'font.serif': 'cmr10',
                                     'axes.formatter.use_mathtext': True})
            # make all plot text same size
            plt.rcParams.update({'font.size': 16})

        if plot_file is None or isinstance(plot_file, str):
            self.plot_file = plot_file
        else:
            raise ValueError(f'wrong `plot_file` parameter: {plot_file}')

        if vf_plot_file is None or isinstance(vf_plot_file, str):
            self.vf_plot_file = vf_plot_file
        else:
            raise ValueError(f'wrong `vf_plot_file` parameter: {vf_plot_file}')

        if self.verbose:
            print(f'PC-ACO initialized successfully at {start_time} with parameters:', flush=True)
            for key, value in self.__dict__.items():
                if isinstance(value, int) or isinstance(value, str) or isinstance(value, bool) \
                        or isinstance(value, float) or isinstance(value, list):
                    print(f'\t{key}:'.ljust(30) + f'\t{value}', flush=True)
                else:
                    print(f'\t{key}:'.ljust(30) + f'\t{value.__class__.__name__}', flush=True)
        self.start_time = start_time
        self.duration = 0.0

        # initializing ant colony
        # using both numpy and random seeding to make sure that the same results are achieved with the same seed,
        # no matter the RNG used by some libraries
        np.random.seed(seed)
        random.seed(seed)
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
        if not self.without_nondominance_ranking:
            non_dominated_fronts = self.nds.do(objective_values)
            ranked_population = np.zeros(self.ants)
            ranking_start = 0
            for front in non_dominated_fronts:
                ranked_front = self.model.rank([objective_values[index] for index in front]) + ranking_start
                for index, rank in zip(front, ranked_front):
                    ranked_population[index] = rank
                ranking_start = np.max(ranked_front) + 1
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

    def update_preference_model(self, objective_values, init=False):
        # randomly select 2 non-dominated solutions and ask DM for comparison
        # for the random selection use the first available front with more than 1 solution (counting from the best one)
        non_dominated_fronts = self.nds.do(objective_values)
        selected_front = None
        for front in non_dominated_fronts:
            if len(front) > 1:
                selected_front = front
                break
        if selected_front is not None:
            selected_index1, selected_index2 = np.random.choice(selected_front, size=2, replace=False)
        else:
            # if there is only 1 solution in every front, then use the solutions from the first and second front
            # they are not non-dominated, but it handles an unlikely theoretically possible edge-case
            selected_index1, selected_index2 = non_dominated_fronts[0][0], non_dominated_fronts[1][0]
        # calculate user value function based on objective values
        obj1, obj2 = objective_values[selected_index1], objective_values[selected_index2]
        val1 = self.user_value_function.calculate(obj1)
        val2 = self.user_value_function.calculate(obj2)
        if val1 <= val2:
            compared_pair = [obj1, obj2]
        else:
            compared_pair = [obj2, obj1]
        self.model.update(compared_pair, init=init)

    def reflected_normal_distribution(self, loc, scale, min_val, max_val):
        # safely get a number from normal distribution
        # in case the value is out of the allowed bounds, reflect the value on the lower/upper bound
        # in case the value is still out of bounds, trim to the lower/upper bound
        base_val = np.random.normal(loc, scale)
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
            k = np.random.choice(range(self.ants), size=1, p=self.aco_probabilities)

            # step 2: randomly sample the value from a parametrized normal distribution
            # calculate std for selected gaussian function
            # based on equation (9) from K. Socha & M. Dorigo, Ant colony optimization for continuous domains, 2008
            std = self.xi * np.sum(np.abs(population[:, n] - population[k, n])) / (self.ants - 1)
            solution[n] = self.reflected_normal_distribution(loc=self.aco_means[k, n], scale=std,
                                                             min_val=self.problem.xl[n],
                                                             max_val=self.problem.xu[n]).item()
        return np.array(solution)

    def save(self, objective_values, actual_generations):
        convergence_indicators = [self.user_value_function.calculate(obj) for obj in objective_values]
        results_df = pd.DataFrame({
            'start_time': self.start_time,
            'seed': self.seed,
            'generations': self.generations,
            'actual_generations': actual_generations,
            'ants': self.ants,
            'q': self.q,
            'xi': self.xi,
            'interval': self.interval,
            'buffer': self.buffer,
            'problem': self.problem.__class__.__name__,
            'variables': self.variables,
            'objectives': self.objectives,
            'user_value_function': str(self.user_value_function),
            'extreme_objective': self.extreme_objective,
            'model': str(self.model),
            'without_nondominance_ranking': self.without_nondominance_ranking,
            'max_no_improvement': self.max_no_improvement,
            'best_convergence': np.min(convergence_indicators),
            'avg_convergence': np.mean(convergence_indicators),
            'duration': self.duration
        }, index=[0])
        if os.path.isfile(self.results_file):
            results_df.to_csv(self.results_file, mode='a', sep=';', index=False, header=False)
        else:
            results_df.to_csv(self.results_file, mode='w', sep=';', index=False, header=True)
        if self.verbose:
            print(f'Results saved to {self.results_file}')

    def plot(self, history):
        if self.objectives == 2:
            plt.figure(figsize=(10, 10))
            try:
                pareto_front = self.problem.pareto_front()
                plt.plot(pareto_front[:, 0], pareto_front[:, 1], color='dimgrey', marker=None, alpha=0.5,
                            label=self.problem.__class__.__name__ + ' Pareto-front', linewidth=3, linestyle='--')
            except:
                print('Pareto-front plotting error.')
            for gen_to_plot in self.plotting_checkpoints:
                if gen_to_plot <= len(history):
                    plt.scatter(history[gen_to_plot-1][:, 0], history[gen_to_plot-1][:, 1],
                                c=self.color_dict[gen_to_plot], marker=self.marker_dict[gen_to_plot], alpha=0.7,
                                label=f'PC-ACO-{str(self.model)} after {gen_to_plot} gen.')
                else:
                    # handle case when pc-aco stops early due to no improvement
                    plt.scatter(history[-1][:, 0], history[-1][:, 1],
                                c=self.color_dict[gen_to_plot], marker=self.marker_dict[gen_to_plot], alpha=0.7,
                                label=f'PC-ACO-{str(self.model)} after {len(history)} gen. (early stop)')
                    break
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            if self.plotting_ticks is not None:
                plt.xticks(self.plotting_ticks['x'])
                plt.yticks(self.plotting_ticks['y'])
            plt.grid('both')
            if self.problem.__class__.__name__ in ['WFG4', 'WFG9']:
                # move legend to lower left corner for concave problems
                plt.legend(loc='lower left')
            else:
                plt.legend(loc='upper right')
            if self.plot_file is None:
                file = f'results/plot_{self.start_time}' + ('.pdf' if self.pdf_plots else '.png')
            else:
                file = self.plot_file
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            if self.verbose:
                print(f'Plot saved to {file}')
        else:
            print('Plotting error - only instances with 2 objectives can be plotted.', flush=True)

    def show_final_value_function(self, vf_history):
        plt.figure(figsize=(10 * self.objectives, 10))
        for obj in range(self.objectives):
            max_util = -np.inf
            plt.subplot(1, self.objectives, obj+1)
            if str(self.model) == 'MC':
                for gen_to_plot in self.plotting_checkpoints:
                    if gen_to_plot <= len(vf_history):
                        for vf in range(len(vf_history[gen_to_plot-1])):
                            plt.plot(vf_history[gen_to_plot-1][vf][obj]['obj'],
                                     vf_history[gen_to_plot-1][vf][obj]['util'],
                                     color=self.color_dict[gen_to_plot], marker=self.marker_dict[gen_to_plot],
                                     alpha=0.7, linewidth=3,
                                     label=(f'PC-ACO-{str(self.model)} after {gen_to_plot} gen.' if vf == 0 else None))
                            tmp_max_util = np.max(vf_history[gen_to_plot-1][vf][obj]['util'])
                            if tmp_max_util > max_util:
                                max_util = tmp_max_util
                    else:
                        for vf in range(len(vf_history[-1])):
                            plt.plot(vf_history[-1][vf][obj]['obj'], vf_history[-1][vf][obj]['util'],
                                     color=self.color_dict[gen_to_plot], marker=self.marker_dict[gen_to_plot],
                                     alpha=0.7, linewidth=3,
                                     label=((f'PC-ACO-{str(self.model)} after {len(vf_history)} gen. '
                                             f'(early stop)') if vf == 0 else None))
                            tmp_max_util = np.max(vf_history[-1][vf][obj]['util'])
                            if tmp_max_util > max_util:
                                max_util = tmp_max_util
                        break
            else:
                for gen_to_plot in self.plotting_checkpoints:
                    if gen_to_plot <= len(vf_history):
                        plt.plot(vf_history[gen_to_plot-1][obj]['obj'], vf_history[gen_to_plot-1][obj]['util'],
                                 color=self.color_dict[gen_to_plot], marker=self.marker_dict[gen_to_plot],
                                 alpha=0.7, linewidth=3,
                                 label=f'PC-ACO-{str(self.model)} after {gen_to_plot} gen.')
                        tmp_max_util = np.max(vf_history[gen_to_plot-1][obj]['util'])
                        if tmp_max_util > max_util:
                            max_util = tmp_max_util
                    else:
                        # handle case when pc-aco stops early due to no improvement
                        plt.plot(vf_history[-1][obj]['obj'], vf_history[-1][obj]['util'],
                                 color=self.color_dict[gen_to_plot], marker=self.marker_dict[gen_to_plot],
                                 alpha=0.7, linewidth=3,
                                 label=f'PC-ACO-{str(self.model)} after {len(vf_history)} gen. (early stop)')
                        tmp_max_util = np.max(vf_history[-1][obj]['util'])
                        if tmp_max_util > max_util:
                            max_util = tmp_max_util
                        break
            plt.xlabel(f'Objective {obj+1}')
            plt.ylabel(f'Marginal value of objective {obj+1}')
            plt.xticks(np.arange(self.model.best_obj[obj], np.round(self.model.worst_obj[obj]+0.5, 1), 0.5))
            if max_util < 0.01:
                # special handling for very low values of utility
                plt.yticks(np.linspace(0.0, max_util, 11))
            else:
                plt.yticks(np.arange(0.0, 1.1, 0.1))
            plt.grid('both')
            plt.legend()  # no fixed legend position, better to do it automatically
        if self.vf_plot_file is None:
            file = f'results/vf_{self.start_time}' + ('.pdf' if self.pdf_plots else '.png')
        else:
            file = self.vf_plot_file
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        if self.verbose:
            print(f'Value function plot saved to {file}')

    def solve(self):
        start = time.perf_counter()
        history = []
        vf_history = []

        # random initialization of the 0th population
        population = np.random.uniform(low=np.tile(self.problem.xl, (self.ants, 1)),
                                       high=np.tile(self.problem.xu, (self.ants, 1)),
                                       size=(self.ants, self.variables))
        objective_values = self.problem.evaluate(population)

        # assign approximate bounds of objective values in preference model (needed for interpolation)
        self.model.best_obj = np.zeros(self.objectives)
        self.model.worst_obj = np.ceil(np.max(objective_values, axis=0))

        self.update_preference_model(objective_values, init=True)
        self.update_ant_colony(population, objective_values)
        history.append(objective_values)
        vf_history.append(self.model.interp_points)

        convergence_indicators = [self.user_value_function.calculate(obj_val) for obj_val in objective_values]
        last_performance = np.round(np.mean(convergence_indicators))
        count_no_improvement = 0
        if self.verbose:
            print(f'Finished generation 0 after {np.round(time.perf_counter() - start, 3)}s from start '
                  f'(best={np.round(np.min(convergence_indicators))}; '
                  f'avg={last_performance}; '
                  f'buffer={len(self.model.buffer)})', flush=True)

        # main ACO loop
        for g in range(1, self.generations+1):
            new_population = np.array([self.construct_solution(population) for _ in range(self.ants)])
            new_objective_values = self.problem.evaluate(new_population)

            # check if the objective upper bound is preserved, update if necessary (needed for interpolation)
            max_obj = np.max(new_objective_values, axis=0)
            for obj in range(self.objectives):
                if max_obj[obj] > self.model.worst_obj[obj]:
                    new_upper_bound = np.ceil(max_obj[obj])
                    if self.verbose:
                        print(f'Upper bound of objective {obj} ({self.model.worst_obj[obj]}) exceeded and updated:',
                              new_upper_bound)
                    self.model.update_upper_bound_on_objective(obj, new_upper_bound)

            # omit updates in the last generation as aco and model are no longer needed
            if g != self.generations:
                # update preference model every 'interval' generations
                if g % self.interval == 0:
                    self.update_preference_model(new_objective_values)
                # update ant colony parameters every generation
                self.update_ant_colony(new_population, new_objective_values)

            population = new_population
            objective_values = new_objective_values
            history.append(objective_values)
            vf_history.append(self.model.interp_points)

            convergence_indicators = [self.user_value_function.calculate(obj_val) for obj_val in objective_values]
            current_performance = np.round(np.mean(convergence_indicators), 3)
            if self.verbose:
                print(f'Finished generation {g} after {np.round(time.perf_counter() - start, 3)}s from start '
                      f'(best={np.round(np.min(convergence_indicators), 3)}; '
                      f'avg={current_performance}; '
                      f'buffer={len(self.model.buffer)}'
                      + ('; no improvement!)' if current_performance >= last_performance else ')'), flush=True)

            # early stopping if 'max_no_improvement' generations without avg. improvement in the first 3 decimal places
            if current_performance >= last_performance:
                count_no_improvement += 1
                if count_no_improvement == self.max_no_improvement:
                    if self.verbose:
                        print(f'Early stopping due to {count_no_improvement} generations without avg. improvement.')
                    break
            else:
                count_no_improvement = 0
            last_performance = current_performance

        stop = time.perf_counter()
        self.duration = stop - start
        if not self.dont_save_csv:
            self.save(objective_values, len(history)-1)
        if self.draw_plot:
            self.plot(history[1:])
        if self.draw_value_function:
            self.show_final_value_function(vf_history[1:])
        if self.verbose:
            print(f'PC-ACO completed optimization successfully in {np.round(self.duration, 3)}s')

        # return best, avg, duration, history
        return np.min(convergence_indicators), np.mean(convergence_indicators), self.duration, history


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generations', type=int, default=100, help='number of ACO generations; default=100')
    parser.add_argument('-a', '--ants', type=int, default=30, help='number of ants and population size; default=30')
    parser.add_argument('-q', type=float, default=0.1, help='diversification parameter of ACO; default=0.1')
    parser.add_argument('-xi', type=float, default=0.5, help='convergence parameter of ACO; default=0.5')
    parser.add_argument('-i', '--interval', type=int, default=10,
                        help='intervals between asking the artificial user for a pairwise comparison; '
                             'interval<generations; default=10')
    parser.add_argument('-b', '--buffer', type=int, default=30,
                        help='max number of stored pairwise comparisons; default=30')
    parser.add_argument('-p', '--problem', default='wfg4',
                        choices=['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7',
                                 'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9',
                                 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt5', 'zdt6'],
                        help='type of MOO benchamark problem to solve (from pymoo library); default=wfg4')
    parser.add_argument('-v', '--variables', type=int, default=10,
                        help='number of variables for DTLZ and WFG problems (not applicable for ZDT); default=10')
    parser.add_argument('-o', '--objectives', type=int, default=2, choices=[2, 3, 4, 5],
                        help='number of objectives (2, 3, 4 or 5) for DTLZ and WFG problems (not applicable for ZDT); '
                             'default=2')
    parser.add_argument('-u', '--user-value-function', choices=['linear', 'chebycheff'], default='linear',
                        help='type of the user`s (decision maker`s) value function for comparisons; default=linear')
    parser.add_argument('-e', '--extreme-objective', action='store_true',
                        help='the user considers one of the objectives as much more important than all others; '
                             'default=False')
    parser.add_argument('-m', '--model', choices=['mdvf', 'mscvf', 'msvf', 'mrvf', 'mc'], default='mdvf',
                        help='type of the value function approach for the preference model; default=mdvf')
    parser.add_argument('-r', '--without-nondominance-ranking', action='store_true',
                        help='turn off the nondominance ranking during solution sorting; default=False')
    parser.add_argument('-mni', '--max-no-improvement', type=int, default=20,
                        help='max number of generations without improvement; default=20')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='random number generator seed; default=42')
    parser.add_argument('-c', '--dont-save-csv', action='store_true',
                        help='turn off saving results to csv; default=False')
    parser.add_argument('-f', '--results-file', type=str, default='results/results.csv',
                        help='path to the CSV file where final results will be saved')
    parser.add_argument('-d', '--draw-plot', action='store_true',
                        help='draw plot of the results; default=False')
    parser.add_argument('-dvf', '--draw-value-function', action='store_true',
                        help='draw plot of the value functions across all generations and objectives; default=False')
    parser.add_argument('-pdf', '--pdf-plots', action='store_true',
                        help='save plots as PDF file; if not enabled, plots will be saved as PNG; default=False')
    parser.add_argument('--verbose', action='store_true',
                        help='print messages to terminal; default=False')
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
                                                          without_nondominance_ranking=args.without_nondominance_ranking,
                                                          seed=args.seed,
                                                          dont_save_csv=args.dont_save_csv,
                                                          results_file=args.results_file,
                                                          draw_plot=args.draw_plot,
                                                          pdf_plots=args.pdf_plots,
                                                          verbose=args.verbose)
    pcaco.solve()
