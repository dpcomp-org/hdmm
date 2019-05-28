from src.hdmm import workload, templates, error


def marginal_index_repr(i, dimensions, join_string=""):
    formatter = '{:0%db}' % dimensions
    return join_string.join(formatter.format(i))


def error_evaluation_marginal(W, A_list, columns=None):
    I = 2**len(W.domain)
    for i in range(I):
        if W.weights[i] > 0:
            print(
                f"{i :4d}", \
                "".join('{:09b}'.format(i)),  # bit representation of marginal, space-delim
                W.weights[i],
                *[A.strategy().weights[i] for A in A_list],
            )



