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
    params['workload'] = '2drangeid-kron'
    params['domain'] = 256
    params['init_scale'] = 1.0

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--workload', choices=['2dunion-kron', '2drangeid-kron', '2dprefixid-kron'])
    parser.add_argument('--domain', type=int, help='domain size n')
    parser.add_argument('--augment', type=int, help='rows of augmented matrix (default n/16)')
    parser.add_argument('--init_scale', type=float, help='scale of initial augmented matrix')
    parser.add_argument('--save', help='path to save strategy to')

    parser.set_defaults(**default_params()) 
    args = parser.parse_args()

    n = args.domain
    p = args.augment

    workloads = {}

    if args.workload == '2dunion-kron':
        W = workload.RangeTotal2D(n)
    if args.workload == '2drangeid-kron':
        W = workload.RangeIdentity2D(n)
    if args.workload == '2dprefixid-kron':
        W = workload.PrefixIdentity2D(n)

    if p is None:
        p = n // 16
    init = [np.random.rand(p,n)*args.init_scale for _ in [0,1]]
    WtWs = [[w2.WtW for w2 in w1.workloads] for w1 in W.workloads]

    ans = optimize.union_kron(WtWs, init)
    
    print W.rootmse(ans['As']), W.rootmse([np.eye(n), np.eye(n)])

#    print 'RMSE', np.sqrt(ans['log'][-1] / W.queries) 

    if args.save:
        pickle.dump(ans, open(args.save, 'wb'))

#    np.save(args.save, A)

