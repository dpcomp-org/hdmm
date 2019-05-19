import pickle
from hdmm import templates, error

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
    return loaded['W'], loaded['domain'], loaded['attr']






if __name__ == '__main__':

    W, domain, attr = dhc_household()

    M = templates.Marginals(domain)

    for i in range(10):
        M.optimize(W)
        print(error.rootmse(W, M.strategy(), eps=.1))
