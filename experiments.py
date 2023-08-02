from pcaco import PairwiseComparisonsBasedAntColonyOptimization


for problem in ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7',
                'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9',
                'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt5', 'zdt6']:

    pcaco = PairwiseComparisonsBasedAntColonyOptimization(generations=100, ants=100, problem=problem,
                                                          variables=10, objectives=2,
                                                          save_csv=True, draw_plot=True, verbose=True)
    pcaco.solve()