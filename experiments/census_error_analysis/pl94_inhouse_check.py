from hdmm.workload import *
from hdmm import templates, error
import itertools


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
    return EkteloMatrix(gqlevels)

def __household():
    household = np.zeros((2,8))
    household[0,0] = 1.0
    household[1,1:] = 1.0
    return EkteloMatrix(household)

def __institutionalized():
    instit = np.zeros((2,8))
    instit[0,1:5] = 1.0
    instit[1,5:8] = 1.0
    return EkteloMatrix(instit)

def __numraces():
    numraces = np.zeros((6, 63))
    for i in range(1, 64):
        ct = bin(i).count('1')  # number of races
        numraces[ct-1, i-1] = 1.0
    return EkteloMatrix(numraces)

def pl94_workload(with_full_id=False):

    cenrace = Kronecker([Total(2), Total(2), Identity(63), Total(8)])
    gqlevels = Kronecker([Total(2), Total(2), Total(63), __gqlevel()])
    hispanic = Kronecker([Identity(2), Total(2), Total(63), Total(8)])
    hispanic_cenrace = Kronecker([Identity(2), Total(2), Identity(63), Total(8)])
    hispanic_numraces = Kronecker([Identity(2), Total(2), __numraces(), Total(8)])
    household = Kronecker([Total(2), Total(2), Total(63), __household()])
    institutionlized = Kronecker([Total(2), Total(2), Total(63), __institutionalized()])
    numraces = Kronecker([Total(2), Total(2), __numraces(), Total(8)])
    total = Kronecker([Total(2), Total(2), Total(63), Total(8)])
    votingage = Kronecker([Total(2), Identity(2), Total(63), Total(8)])
    votingage_cenrace = Kronecker([Total(2), Identity(2), Identity(63), Total(8)])
    votingage_hispanic = Kronecker([Identity(2), Identity(2), Total(63), Total(8)])
    votingage_hispanic_cenrace = Kronecker([Identity(2), Identity(2), Identity(63), Total(8)])
    votingage_hispanic_numraces = Kronecker([Identity(2), Identity(2), __numraces(), Total(8)])
    votingage_numraces = Kronecker([Total(2), Identity(2), __numraces(), Total(8)])

    full_id = Kronecker([Identity(2), Identity(2), Identity(63), Identity(8)])

    W_list = [cenrace, gqlevels, hispanic, hispanic_cenrace, hispanic_numraces, household,
            institutionlized, numraces, total, votingage, votingage_cenrace, votingage_hispanic,
            votingage_hispanic_cenrace, votingage_hispanic_numraces, votingage_numraces]

    if with_full_id:
        W_list.append(full_id)

    return VStack(W_list)


def opt_p_identity(workload=None):
    ps = [1, 1, 8, 4]   # hard-coded parameters
    ns = [2,2,63,8]

    template = templates.KronPIdentity(ps, ns)
    A = template.restart_optimize(workload, 25)[0]
    return A


# build manual strategy
def manual_strategy():
    weights = { (1,1,1,1) : 0.1, (1,1,1,0) : 0.675, (0,0,0,1) : 0.225 }
    #weights = { (0,1,2,3) : 0.1, (0,1,2) : 0.675, (3,) : 0.225 }
    A = Marginals.frombinary((2,2,63,8), weights)
    return A

def marginal_strategy(workload=None):
    
    template = templates.Marginals((2,2,63,8))
    A = template.restart_optimize(workload, 25)[0]
    return A

if __name__ == '__main__':

    W = pl94_workload(with_full_id=True)

    print(W.shape)

    A = opt_p_identity(workload=W)
    err = error.rootmse(W, A)
    print('KroneckerPIdentity', err)

    #W = Marginals.approximate(W)
    A = marginal_strategy(workload=W)
    marg_err = error.rootmse(W, A)
    #marg_err = np.sqrt( error.expected_error(W, A) / (3*3*64*9) )
    print('Marginals', marg_err)

    robert_err = error.rootmse(W, manual_strategy())
    print('Robert', robert_err)

    print(A.weights / A.weights.sum())
