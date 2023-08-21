import argparse
from tqdm import tqdm

from pcaco import PairwiseComparisonsBasedAntColonyOptimization

if __name__ == '__main__':
    # code for conducting the experiments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['mdvf', 'mscvf', 'msvf', 'mrvf', 'mc', 'all'], default='all',
                        help='with which type of the value function approach will the experiments be run; default=all')
    parser.add_argument('-s', '--start_seed', type=int, default=100, help='starting seed')
    parser.add_argument('-n', '--num_seeds', type=int, default=20,
                        help='number of runs (different consecutive seeds from `start_seed`)')
    args = parser.parse_args()

    model_list = None
    if args.model == 'all':
        model_list = ['mdvf', 'mscvf', 'msvf', 'mrvf', 'mc']
    else:
        model_list = [args.model]
    results_file = 'results/' + args.model + '_results.csv'

    problem_list = ['wfg1', 'wfg3', 'wfg4', 'wfg9']
    obj_list = [2, 3, 4, 5]
    uvf_list = ['linear', 'chebycheff']
    extr_list = [False, True]
    mc_interval_list = [10, 20]
    interval_list = [1, 10, 20] if args.model != 'mc' else mc_interval_list

    cases = (len(problem_list) * len(obj_list) * len(uvf_list) * len(extr_list)
             * (len(interval_list) if args.model != 'all' else (4 * len(interval_list) + len(mc_interval_list))))
    print(f'Starting calculation of {cases} cases, {args.num_seeds} runs for each '
          f'({cases * args.num_seeds} runs in total)')
    count = 0

    for problem in problem_list:
        gens = 100 if problem != 'wfg1' else 300
        for obj in obj_list:
            for uvf in uvf_list:
                for extr in extr_list:
                    for model in model_list:
                        intervals = interval_list if model != 'mc' else mc_interval_list
                        for interval in intervals:
                            count += 1
                            print(f'Starting case {count}/{cases}: problem={problem}; obj={obj}; uvf={uvf}; extr={extr}; '
                                  f'model={model}; i={interval}', flush=True)
                            for seed in tqdm(range(args.start_seed, args.start_seed + args.num_seeds)):
                                pcaco = PairwiseComparisonsBasedAntColonyOptimization(generations=gens,
                                                                                      ants=30,
                                                                                      problem=problem,
                                                                                      variables=10,
                                                                                      objectives=obj,
                                                                                      interval=interval,
                                                                                      user_value_function=uvf,
                                                                                      extreme_objective=extr,
                                                                                      model=model,
                                                                                      without_nondominance_ranking=False,
                                                                                      max_no_improvement=20,
                                                                                      dont_save_csv=False,
                                                                                      draw_plot=False,
                                                                                      draw_value_function=False,
                                                                                      pdf_plots=False,
                                                                                      verbose=False,
                                                                                      seed=seed,
                                                                                      plotting_checkpoints=None,
                                                                                      plotting_ticks=None,
                                                                                      results_file=results_file)
                                pcaco.solve()
