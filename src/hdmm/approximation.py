import numpy as np
import workload
import itertools

def marginals_approx(W):
    """
    Given a Union-of-Kron workload, find a Marginals workload that approximates it.
    
    The guarantee is that for all marginals strategies A, Error(W, A) = Error(M, A) where
    M is the returned marginals approximation of W.

    The other guarantee is that this function is idempotent: approx(approx(W)) = approx(W)
    """
    if isinstance(W, workload.Kron):
        W = workload.Concat([W])
    assert isinstance(W, workload.Concat) and isinstance(W.workloads[0], workload.Kron)
    dom = W.domain
    weights = np.zeros(2**len(dom))
    for sub in W.workloads:
        tmp = []
        for n, piece in zip(dom, sub.workloads):
            X = piece.WtW
            b = float(X.sum() - X.trace()) / (n * (n-1))
            a = float(X.trace()) / n - b
            tmp.append(np.array([b,a]))
        weights += reduce(np.kron, tmp)
    keys = itertools.product(*[[0,1]]*len(dom))
    weights = dict(zip(keys, np.sqrt(weights)))
    return workload.Marginals(dom, weights) 
 
if __name__ == '__main__':
    from experiments.census_workloads import *
    import optimize
    import implicit

    sf1 = CensusSF1()
    approx = marginals_approx(sf1)

    eye = [np.eye(n) for n in sf1.domain]
    eye[-1] *= 5
    print(sf1.expected_error(eye), workload.Concat.expected_error(approx, eye))

    ans = optimize.optimize_marginals(sf1.domain, approx.weight_vector())
    A1 = implicit.marginals_inverse(sf1.domain, ans['theta'], ans['invtheta'])
    noise = lambda: np.random.laplace(loc=0, scale=1.0/np.sqrt(2), size=A1.shape[1])
    noises = [A1.dot(noise()) for _ in range(100)]
    print(ans['error'])
    print(approx.average_error_ci(noises))
    print(sf1.average_error_ci(noises))
