import numpy as np
import matplotlib.pyplot as plt

from pcaco import PairwiseComparisonsBasedAntColonyOptimization

if __name__ == '__main__':
    # code for generating generation plots for reports
    ticks = {'wfg1': {'x': np.round(np.arange(0.0, 3.5, 0.5), 2),
                      'y': np.round(np.arange(0.0, 4.5, 0.5), 2)},
             'wfg3': {'x': np.round(np.arange(0.0, 3.0, 0.5), 2),
                      'y': np.round(np.arange(0.0, 5.0, 0.5), 2)},
             'wfg4': {'x': np.round(np.arange(0.0, 3.0, 0.5), 2),
                      'y': np.round(np.arange(0.0, 5.0, 0.5), 2)},
             'wfg9': {'x': np.round(np.arange(0.0, 3.0, 0.5), 2),
                      'y': np.round(np.arange(0.0, 5.0, 0.5), 2)},
             }
    problem = 'wfg1'
    gens = 100 if problem != 'wfg1' else 300
    plt_chkpts = [10, 33, 67, 100] if problem != 'wfg1' else [30, 100, 200, 300]

    # convergence plot, generation plot and value function comparison for a single problem
    plt.figure(figsize=(10, 10))
    colors = ['tab:orange', 'navy', 'tab:cyan', 'tab:olive', 'firebrick']
    for m, model in enumerate(['mdvf', 'mscvf', 'msvf', 'mrvf', 'mc']):
        pcaco = PairwiseComparisonsBasedAntColonyOptimization(generations=gens,
                                                              ants=30,
                                                              problem=problem,
                                                              variables=10,
                                                              objectives=2,
                                                              interval=10,
                                                              user_value_function='linear',
                                                              extreme_objective=False,
                                                              model=model,
                                                              without_nondominance_ranking=False,
                                                              max_no_improvement=20,
                                                              dont_save_csv=False,
                                                              draw_plot=True,
                                                              draw_value_function=True,
                                                              pdf_plots=True,
                                                              verbose=True,
                                                              seed=987654321,
                                                              plotting_checkpoints=plt_chkpts,
                                                              plotting_ticks=ticks[problem],
                                                              plot_file=f'results/plot_{model}.pdf',
                                                              vf_plot_file=f'results/vf_{model}.pdf')
        _, _, _, history = pcaco.solve()
        convergence_indicators = [np.min([pcaco.user_value_function.calculate(obj_val) for obj_val in generation])
                                  for generation in history]
        plt.plot(range(0, len(history)), convergence_indicators,
                 color=colors[m], marker=None, alpha=0.7, linewidth=2, linestyle='solid',
                 label=f'PC-ACO-{str(pcaco.model)}')
        # mark the final generation
        plt.scatter([len(history)-1], [convergence_indicators[-1]], color=colors[m], marker='x', alpha=0.7, zorder=5)
    plt.xlabel(f'Generations')
    plt.ylabel(f'Best performance in the current generation')
    plt.xticks(range(0, gens+1, 50))
    plt.yticks(np.arange(1.1, 2.3, 0.1))
    plt.grid('both')
    plt.legend(loc='upper right')
    plt.savefig(f'results/{problem}_convergence_plot.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # show a single case with/without nondominance ranking
    problem = 'wfg9'
    gens = 100 if problem != 'wfg1' else 300
    plt_chkpts = [10, 33, 67, 100] if problem != 'wfg1' else [30, 100, 200, 300]
    plt.figure(figsize=(10, 10))
    colors = ['tab:orange', 'navy']
    for w, without_ndr in enumerate([False, True]):
        with_or_without = 'without' if without_ndr else 'with'
        ranking_type = 'one-tier' if without_ndr else 'two-tier'
        pcaco = PairwiseComparisonsBasedAntColonyOptimization(generations=gens,
                                                              ants=30,
                                                              problem=problem,
                                                              variables=10,
                                                              objectives=2,
                                                              interval=10,
                                                              user_value_function='linear',
                                                              extreme_objective=False,
                                                              model='mdvf',
                                                              without_nondominance_ranking=without_ndr,
                                                              max_no_improvement=20,
                                                              dont_save_csv=False,
                                                              draw_plot=True,
                                                              draw_value_function=False,
                                                              pdf_plots=True,
                                                              verbose=True,
                                                              seed=4321,
                                                              plotting_checkpoints=plt_chkpts,
                                                              plotting_ticks=ticks[problem],
                                                              plot_file=f'results/plot_{with_or_without}_ndr.pdf')
        _, _, _, history = pcaco.solve()
        convergence_indicators = [np.min([pcaco.user_value_function.calculate(obj_val) for obj_val in generation])
                                  for generation in history]
        plt.plot(range(0, len(history)), convergence_indicators,
                 color=colors[w], marker=None, alpha=0.7, linewidth=2, linestyle='solid',
                 label=f'PC-ACO-MDVF {ranking_type} ranking')
        # mark the final generation
        plt.scatter([len(history) - 1], [convergence_indicators[-1]], color=colors[w], marker='x', alpha=0.7, zorder=5)
    plt.xlabel(f'Generations')
    plt.ylabel(f'Best performance in the current generation')
    plt.xticks(range(0, gens + 1, 20))
    plt.yticks(np.arange(1.2, 2.2, 0.1))
    plt.grid('both')
    plt.legend(loc='upper right')
    plt.savefig(f'results/{problem}_ndr_convergence_plot.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()
