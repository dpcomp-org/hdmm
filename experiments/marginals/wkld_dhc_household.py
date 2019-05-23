import pickle
from hdmm import templates, error, workload
from experiments.marginals import util
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


if __name__ == '__main__':

    W, domain = dhc_household()
    seed = 0

    # Define strategies
    A_marg = templates.Marginals(domain, seed=seed)
    A_marg.optimize(W)

    W_approx = workload.Marginals.approximate(W)

    A_identity = templates.Marginals(domain)
    A_identity._params = np.zeros(2**len(domain))
    A_identity._params[-1] = 1.0

    print('Num queries:', W.shape[0])
    print('Sensitivity:', W.sensitivity())
    print('Marg', '\t\t', f'{error.rootmse(W, A_marg.strategy()):10.3f}')
    print('Ident', '\t\t', f'{error.rootmse(W, A_identity.strategy()):10.3f}')
    print('W_approx', '\t', f'{error.rootmse(W, W_approx):10.3f}')

    # summarize_strategy(W_approx, A_marg, domain)

    print('')

    for m in W.matrices:
        print(m, '\t', error.rootmse(m, A_marg.strategy()))

