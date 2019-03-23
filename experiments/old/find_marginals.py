import argparse
import pandas as pd
import numpy as np
from IPython import embed
import workload
import optimize
import pickle

"""
This is a simple description of functionality of this program.
"""

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['workload'] = 'lowd'
    params['n'] = 10
    params['d'] = 8
    params['k'] = 4

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--workload', choices=['lowd', 'kway'], help='workload type')
    parser.add_argument('--n', type=int, help='domain size of each dimensions')
    parser.add_argument('--d', type=int, help='number of dimensions')
    parser.add_argument('--k', type=int, help='largest marginal table in workload')
    parser.add_argument('--save', help='path to save results to')

    parser.set_defaults(**default_params()) 
    args = parser.parse_args()

    dom = [args.n] * args.d

    if args.workload == 'lowd':
        W = workload.DimKMarginals(dom, range(args.k+1))
    elif args.workload == 'kway':
        W = workload.DimKMarginals(dom, k)
    
    weights = W.weight_vector()
    ans = optimize.optimize_marginals(dom, weights)

    err = ans['error']
    eye = ans['identity']
    work = ans['workload']
       
    print ans['valid'], np.sqrt(eye/err), np.sqrt(work/err)

    if args.save:
        pickle.dump(ans, open(args.save, 'wb'))
