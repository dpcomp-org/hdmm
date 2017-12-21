import numpy as np
from workload import AllRange
from optimize import augmented_optimization
import time
import pandas as pd
import itertools
from IPython import embed
import matplotlib
#matplotlib.use('Agg')
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
        domain, _ = setting
        R = AllRange(domain)
        B0 = np.random.rand(domain//16, domain)
        t0 = time.time()
        ans = augmented_optimization(R.WtW, B0)
        t1 = time.time()
        err = np.sqrt(ans['res'].fun / R.queries)
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
    params['domains'] = [2**k for k in range(6, 10)]
    params['trials'] = 2
    params['style'] = 'seaborn-paper'
    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--domains', type=int, nargs='+', help='domain sizes')
    parser.add_argument('--trials', type=int, help='number of trials')
    parser.add_argument('--load', help='path to load results from')
    parser.add_argument('--save', help='path to save results to')
    parser.add_argument('--plot', action='store_true', help='plot the times')
    parser.add_argument('--style', choices=plt.style.available, help='plot style')

    parser.set_defaults(**default_params()) 
    args = parser.parse_args()

    trials = range(args.trials)
    domains = args.domains

    names = ['domain', 'trial']
    settings = get_all_settings(domains, trials)
    path = None
    if args.load:
        df = pd.read_csv(args.load, index_col=names)
        path = args.load[:-4]
    else:    
        df = run_experiments(names, settings) 
    
    if args.save:
        path = args.save[:-4]
        df.to_csv(args.save)

    print df.error.unstack('domain').describe()
    print df.time.unstack('domain').describe()

    if args.plot:
        groups = df.groupby(level=0)
        means = groups.time.mean()
        stds = groups.time.std()

        plt.style.use(args.style)
        means.plot(loglog=True, legend=False, style='o-')
        plt.xticks([1e2, 1e3, 1e4])
        plt.xlabel('Domain Size')
        plt.ylabel('Time (s)')
        if path:
            plt.savefig('%s.png' % path)
            plt.savefig('%s.pdf' % path)
        plt.show()

