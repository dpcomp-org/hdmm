from workload import *
import templates
import itertools
from approximation import marginals_approx


# Resources
# https://github.com/dpcomp-org/das_decennial/blob/master/programs/schema/schemas/Schema_PL94.py
# https://github.com/dpcomp-org/das_decennial/blob/master/hdmm_examples/pl94_comparison.py


# Predicates for defining Robert's version of PL94

# Schema is:
# Hispanic(2) Voting-Age(2) Race(63) hhgq(8)

# def __race1():
#     # single race only, two or more races aggregated
#     # binary encoding: 1 indicates particular race is checked
#     race1 = np.zeros((7, 64))
#     for i in range(6):
#         race1[i, 2**i] = 1.0
#     race1[6,:] = 1.0 - race1[0:6].sum(axis=0)
#     return Matrix(race1)
#
# def __race2():
#     # all settings of race, k races for 1..6, two or more races
#     race2 = np.zeros((63+6+1, 64))
#     for i in range(1,64):
#         race2[i-1,i] = 1.0
#         ct = bin(i).count('1') # number of races
#         race2[62+ct, i] = 1.0
#     race2[63+6] = race2[64:63+6].sum(axis=0) # two or more races
#     return Matrix(race2)

def __gqlevel():
    gqlevels = np.zeros((7,8))
    for i in range(7):
        gqlevels[i,i+1] = 1.0
    return Matrix(gqlevels)

def __household():
    household = np.zeros((2,8))
    household[0,0] = 1.0
    household[1,1:] = 1.0
    return Matrix(household)

def __institutionalized():
    instit = np.zeros((2,8))
    instit[0,1:5] = 1.0
    instit[1,5:8] = 1.0
    return Matrix(instit)

def __numraces():
    numraces = np.zeros((6, 63))
    for i in range(1, 64):
        ct = bin(i).count('1')  # number of races
        numraces[ct-1, i-1] = 1.0
    return Matrix(numraces)

cenrace = Kron([Total(2), Total(2), Identity(63), Total(8)])
gqlevels = Kron([Total(2), Total(2), Total(63), __gqlevel()])
hispanic = Kron([Identity(2), Total(2), Total(63), Total(8)])
hispanic_cenrace = Kron([Identity(2), Total(2), Identity(63), Total(8)])
hispanic_numraces = Kron([Identity(2), Total(2), __numraces(), Total(8)])
household = Kron([Total(2), Total(2), Total(63), __household()])
institutionlized = Kron([Total(2), Total(2), Total(63), __institutionalized()])
numraces = Kron([Total(2), Total(2), __numraces(), Total(8)])
total = Kron([Total(2), Total(2), Total(63), Total(8)])
votingage = Kron([Total(2), Identity(2), Total(63), Total(8)])
votingage_cenrace = Kron([Total(2), Identity(2), Identity(63), Total(8)])
votingage_hispanic = Kron([Identity(2), Identity(2), Total(63), Total(8)])
votingage_hispanic_cenrace = Kron([Identity(2), Identity(2), Identity(63), Total(8)])
votingage_hispanic_numraces = Kron([Identity(2), Identity(2), __numraces(), Total(8)])
votingage_numraces = Kron([Total(2), Identity(2), __numraces(), Total(8)])


W = Concat([cenrace, gqlevels, hispanic, hispanic_cenrace, hispanic_numraces, household,
        institutionlized, numraces, total, votingage, votingage_cenrace, votingage_hispanic,
        votingage_hispanic_cenrace, votingage_hispanic_numraces, votingage_numraces])

def opt_p_identity(workload=None):
    ps = [1, 1, 8, 4]   # hard-coded parameters

    best = np.inf
    best_template = None

    for _ in range(25):
        template = templates.KronPIdentity(workload.domain, ps)
        res = template.optimize(workload)
        if res['loss'] < best:
            best = res['loss']
            best_template = template 
    return [sub.A for sub in best_template.strategies]


# build manual strategy
def manual_strategy():
    identity = Kron([Identity(2), Identity(2), Identity(63), Identity(8)]) # weight should be .1
    hhgq = Kron([Total(2), Total(2), Total(63), Identity(8)])   # weight should be .225
    others = Kron([Identity(2), Identity(2), Identity(63), Total(8)])   # weight should be .675

    weights = np.zeros(16)
    weights[15] = 0.1 # binary 1111
    weights[8] = 0.225 # binary 1000
    weights[7] = 0.675 # binary 0111

    # add weights to below:
    return weights

def marginal_strategy(workload=None):
    best, best_template = np.inf, None
    for _ in range(25):
        template = templates.Marginals(workload.domain)
        res = template.optimize(workload)
        if res['loss'] < best:
            best = res['loss']
            best_template = template 
    return best_template.theta

def union_kron():
    W1 = Concat([cenrace, hispanic, hispanic_cenrace, hispanic_numraces,
        numraces, total, votingage, votingage_cenrace, votingage_hispanic,
        votingage_hispanic_cenrace, votingage_hispanic_numraces, votingage_numraces])

    ps = [1,1,8,1]

    best = np.inf
    best_template = None

    for _ in range(25):
        template = templates.KronPIdentity(W1.domain, ps)
        res = template.optimize(W1)
        if res['loss'] < best:
            best = res['loss']
            best_template = template 

    identity = [np.eye(2), np.eye(2), np.eye(63), np.ones((1,8))]
    strategy = [sub.A for sub in best_template.strategies]

    print(W1.rootmse(identity), W1.rootmse(strategy))

    #W2 = Concat([gqlevels, institutionalized, household])


if __name__ == '__main__':

    print(W.queries)

    union_kron()

    A = opt_p_identity(workload=W)
    err = W.rootmse(strategy=A)
    print('KronPIdentity', err)

    W = marginals_approx(W)

    A = marginal_strategy(workload=W)
    marg_err = W.rootmse(strategy=A)
    print('Marginals', marg_err)

    robert_err = W.rootmse(strategy=manual_strategy())
    print('Robert', robert_err)

    print(A / A.sum())

