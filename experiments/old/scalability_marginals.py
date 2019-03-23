import numpy as np
from workload import AllRange, LowDimMarginals
import optimize
#from optimize import datacubes_optimization
#from strategy_opt import augmented_optimization, datacubes_optimization
import time
import pandas as pd
import itertools
from IPython import embed
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def run_experiments(names, params):
    """
    Run experiments and return results as a pandas data frame

    :param names: a list of names of the parameters
    :param params: a list of parameter settings to run experiments for
    :returns: a pandas dataframe with results

    """
    index = pd.MultiIndex.from_tuples(params, names=names)
    results = pd.DataFrame(index=index, columns=['error', 'time'], dtype=float)
    for setting in params:
        print setting
        dims, _ = setting
        domain = [10] * dims
        workload = LowDimMarginals(domain, (dims+1)//2)
        w0 = np.random.rand(2**dims)
        t0 = time.time()
#        res, _ = datacubes_optimization(workload, w0)
        ans = optimize.optimize_marginals(domain, workload.weight_vector())
#        ans = datacubes_optimization(workload)
        t1 = time.time()
        err = np.sqrt(ans['error'] / workload.queries)
        results.loc[setting] = [err, t1 - t0]
    return results

def get_all_settings(*params):
    """
    Create the experiment tuples to run from the range of experiment settings
    over each individual parameter
    :param param1: parameter settings for first parameter (can be a list or a single item)
    :param param2: parameter settings for second parameter (can be a list or a single item)
    :param ...: ...
    :return: a list of parameter setting tuples

    >>> get_all_settings([1,2,3], True, ['x', 'y'])
    [(1, True, 'x'),
     (1, True, 'y'),
     (2, True, 'x'),
     (2, True, 'y'),
     (3, True, 'x'),
     (3, True, 'y')] 
    """
    params2 = [p if type(p) is list else [p] for p in params]
    return list(itertools.product(*params2))

def default_params():
    params = {}
    params['dims'] = range(2, 8)
    params['trials'] = 2
    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dims', type=int, nargs='+', help='list of # dimensions')
    parser.add_argument('--trials', type=int, help='number of trials')
    parser.add_argument('--load', help='path to load results from')
    parser.add_argument('--save', help='path to save results to')
    parser.add_argument('--plot', action='store_true', help='plot the times')

    parser.set_defaults(**default_params()) 
    args = parser.parse_args()

    trials = range(args.trials)
    dims = args.dims

    names = ['dims', 'trial']
    settings = get_all_settings(dims, trials)
    path = None
    if args.load:
        df = pd.read_csv(args.load, index_col=names)
        path = args.load[:-4]
    else:    
        df = run_experiments(names, settings) 
    
    if args.save:
        path = args.save[:-4]
        df.to_csv(args.save)

    print df.error.unstack('dims').describe()
    print df.time.unstack('dims').describe()

    if args.plot:
        groups = df.groupby(level=0)
        means = groups.time.mean()
        stds = groups.time.std()

        means.plot(style='.-g', logy=True, legend=False)
        plt.xlabel('# Dimensions')
        plt.ylabel('Time (s)')
        if path:
            plt.savefig('%s.png' % path)
            plt.savefig('%s.pdf' % path)
        plt.show()

