import pickle
from hdmm import templates, error, workload
from experiments.workload_analysis import util
import os

from ektelo.client import inference
from ektelo.private import measurement

import numpy as np

'''

The DHC Household workload is a complex workload encoded in the Census repo.

This is convenience function that loads the workload from a pickle file which was created from Census code

pickled object is a dict created with this command:

output_dict = {
        'W': W,
        'domain': schema.shape,
        'attr': schema.dimnames
    }


'''


def dhc_household():

    file = open('dhc_household.pckl', 'rb')
    loaded = pickle.load(file)
    return loaded['W'], loaded['domain']


def summarize_strategy(W, A, domain):
    for i in range(2**len(domain)):
        if W.weights[i] > 0 or A._params[i] > 0:
            print(
                i,
                util.marginal_index_repr(i,len(domain), join_string=' '),
                W.weights[i] / sum(W.weights),  # normalized weights of workload
                A._params[i] / sum(A._params)   # normalized weights of strategy
            )


def single_marginal_strategy(key, wgt, domain):
    weights = np.zeros(2 ** len(domain))
    weights[key] = wgt
    return workload.Marginal(domain, key)


def compute_error_matrix(W, A):
    # compute error of each subworkload Wi of W wrt to each sub-strategy Aj
    # if Wi is not supported by Aj, entry (i,j) == inf

    if os.path.isfile('error_matrix.pckl'):
        file = open('error_matrix.pckl', 'r')
        return pickle.load(file)
    else:
        file = open('error_matrix.pckl', 'wb')
        error_matrix = np.zeros((len(W.matrices), len(A.matrices)))
        for i, Wi in enumerate(W.matrices):
            print(i)
            for j, Aj in enumerate(A.matrices):
                if error.strategy_supports_workload(W, A):
                    error_matrix[i, j] = error.expected_error(Wi, Aj.base)
                else:
                    error_matrix[i, j] = float('inf')
            print(error_matrix[i,:])
        pickle.dump(error_matrix, file)
        return error_matrix


def grouped_workload(W, A, error_matrix, match_type='BEST'):
    #
    # Comment
    #

    def best(i, j):  # partition workloads into groups assoc. with best supporting strategy
        return j == np.argmin(error_matrix, axis=1)[i]

    def supported(i, j): # form redundant groups based on all supporting strategies
        return error_matrix[i,j] < float('inf')

    def top(i, j, lim=2):
        return supported(i,j) and (j in np.argsort(error_matrix, axis=1)[0:lim])

    if match_type == 'BEST':
        match = best
    elif match_type == 'TOP':
        match = top
    elif match_type == 'SUPPORTED':
        match = supported
    else:
        assert False

    groups = [[] for _ in range(len(A.matrices))]

    for i in range(len(W.matrices)):
        for j in range(len(A.matrices)):
            if match(i,j):
                groups[j].append(W.matrices[i])

    w_grouped = [workload.VStack(g) for g in groups if len(g) > 0]

    return w_grouped


def expected_error_empirical(W, A, eps=np.sqrt(2), trials=25):
    prng = np.random.RandomState(9999)
    x = np.ones(W.shape[1])     # makeup of data vector doesn't matter
    true_ans = W @ x

    errors = np.zeros(trials)
    for i in range(trials):
        y = measurement.Laplace(A, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(A, y)
        ans = W @ x_hat
        total_sqerr_workload = np.sum((ans-true_ans)**2)
        errors[i] = total_sqerr_workload
    return np.mean(errors)

if __name__ == '__main__':

    W, domain = dhc_household()
    seed = 1008

    # Define strategies
    A_marg = templates.Marginals(domain, seed=seed)
    A_marg.optimize(W)

    # err = expected_error_empirical(W, A_marg.strategy(), trials=50)
    # print(err)
    #
    # print(error.expected_error(W, A_marg.strategy()))

    W_approx = workload.Marginals.approximate(W)

    A_identity = templates.Marginals(domain)
    A_identity._params = np.zeros(2**len(domain))
    A_identity._params[-1] = 1.0

    print('Num queries:', W.shape[0])
    print('Domain size:', W.shape[1])
    print('Sensitivity:', W.sensitivity())
    print('Marg', '\t\t', f'{error.rootmse(W, A_marg.strategy()):10.3f}')
    print('Ident', '\t\t', f'{error.rootmse(W, A_identity.strategy()):10.3f}')
    print('W_approx', '\t', f'{error.rootmse(W, W_approx):10.3f}')

    # summarize_strategy(W_approx, A_marg, domain)

    print('')

    error_matrix = compute_error_matrix(W, A_marg.strategy())

    w_grouped = grouped_workload(W, A_marg.strategy(), error_matrix, match_type='BEST')

    opt_plus = templates.DefaultUnionKron(domain, len(w_grouped))

    B, loss = opt_plus.restart_optimize(w_grouped, 5)

    print(loss)
    print('RMSE', np.sqrt(loss / W.shape[0]))

    # err = expected_error_empirical(W, B, trials=1)
    # print(err)
    # print(np.sqrt(loss / W.shape[0]))


#    supported_subworkload(W, A_marg, domain)

    # for m in W.matrices:
    #     print(m, '\t', error.rootmse(m, A_marg.strategy()))

