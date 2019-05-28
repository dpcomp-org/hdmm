import numpy as np
from workload import *
import templates
import itertools
from examples.census_workloads import build_workload
import seaborn as sns
import matplotlib.pyplot as plt


def opt_strategy(workload=None):
    ps = [10, 10]   # hard-coded parameters
    template = templates.KronPIdentity(workload.domain, ps)
    #template = templates.Marginals(workload.domain)
    template.optimize(workload)
    return [sub.A for sub in template.strategies]


if __name__ == '__main__':

    d = 100   # dimension size
    T = Kron([Total(d), Total(d)])
    m1 = Kron([Identity(d), Total(d)])
    m2 = Kron([Total(d), Identity(d)])
    I = Kron([Identity(d), Identity(d)])

    W = {}
    W['TI'] = Concat([T,I])
    W['T1I'] = Concat([T, m1, I])
    W['T2I'] = Concat([T, m2, I])
    W['T12I'] = Concat([T, m1, m2, I])

    A = {}
    A['TI'] = opt_strategy(W['TI'])
    A['T1I'] = opt_strategy(W['T1I'])
    A['T12I'] = opt_strategy(W['T12I'])

    print(A['TI'])

    print(A['T12I'])

    exit()

    for (a,w) in itertools.product( A.keys(), W.keys()):
        print('w =',w,'\t\t', 'a =',a, '\t', W[w].rootmse(A[a], eps=1) )

