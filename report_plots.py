import numpy as np

from pcaco import PairwiseComparisonsBasedAntColonyOptimization

if __name__ == '__main__':
    # code for generating plots for reports
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

    for model in ['mdvf', 'mscvf', 'msvf', 'ror', 'mc']:
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
                                                              max_no_improvement=300,
                                                              dont_save_csv=False,
                                                              draw_plot=True,
                                                              draw_value_function=True,
                                                              pdf_plots=True,
                                                              verbose=True,
                                                              seed=1234,
                                                              plotting_checkpoints=plt_chkpts,
                                                              plotting_ticks=ticks[problem])
        pcaco.solve()
