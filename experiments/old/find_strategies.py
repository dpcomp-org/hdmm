import argparse
import pandas as pd
import numpy as np
from IPython import embed
import workload
import optimize
import pickle

"""
Find strategy to optimize matrix mechanism error
"""

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['workload'] = 'all-range'
    params['domain'] = 256
    params['init_scale'] = 1.0

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--workload', choices=['all-range', 'prefix', 'permuted-range', 'width25', '2drange', '2dprefix', '2dunion', '2drangeid'], help='workload to optimize')
    parser.add_argument('--domain', type=int, help='domain size n')
    parser.add_argument('--augment', type=int, help='rows of augmented matrix (default n/16)')
    parser.add_argument('--init_scale', type=float, help='scale of initial augmented matrix')
    parser.add_argument('--save', help='path to save strategy to')

    parser.set_defaults(**default_params()) 
    args = parser.parse_args()

    n = args.domain
    p = args.augment

    if args.workload == 'all-range':
        W = workload.AllRange(n)
        WtW = W.WtW
    if args.workload == 'prefix':
        W = workload.Prefix(n)
        WtW = W.WtW
    if args.workload == 'permuted-range':
        idx = np.random.RandomState(0).permutation(n)
        W = workload.AllRange(n)
        WtW = W.WtW[idx,:][:,idx]
    if args.workload == 'width25':
        W = workload.WidthKRange(n, 25)
        WtW = W.WtW
    if args.workload == '2drange':
        W = workload.AllRange(n)
        W = workload.Kron([W, W])
        WtW = W.WtW
        n = n**2
    if args.workload == '2dprefix':
        W = workload.Prefix(n)
        W = workload.Kron([W, W])
        WtW = W.WtW
        n = n**2
    if args.workload == '2dunion':
        R = workload.AllRange(n)
        T = workload.Total(n)
        W = workload.Concat([workload.Kron([R,T]), workload.Kron([T,R])])
        WtW = W.WtW
        if p is None:
            p = 2*n + n//8
        n = n**2
    if args.workload == '2drangeid':
        R = workload.AllRange(n)
        I = workload.Identity(n)
        W = workload.Concat([workload.Kron([R,I]), workload.Kron([I, R])])
        WtW = W.WtW
        n = n**2

    if p is None:
        p = n // 16
    B0 = np.random.rand(p,n) * args.init_scale

    ans = optimize.augmented_optimization(WtW, B0)

    print 'RMSE', np.sqrt(ans['res'].fun / W.queries) 

    if args.save:
        pickle.dump(ans, open(args.save, 'wb'))

#    np.save(args.save, A)

