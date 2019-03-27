import numpy as np
import random
from IPython import embed
from scipy.misc import comb as nCr
from pl94_inhouse_check import pl94_workload, marginal_strategy, manual_strategy, opt_p_identity
from hdmm import inference, error
import matplotlib.pyplot as plt

def synthetic_data(N = 100, seed=111):
    prng = np.random.RandomState(seed)

    # hispanic, voting age, race, hhgq
    x = np.zeros((2,2,63,8)).flatten()

    # database size to sample
    n = 1000000

    P1 = np.array([0.2, 0.8]).reshape(2,1,1,1)
    P2 = np.array([0.8, 0.2]).reshape(1,2,1,1)
    P3 = np.zeros(63)
    lookup = [None, 0.7069, 0.25, 0.04, 0.002, 0.001, 0.0001]
    for race in range(1,64):
        num = bin(race).count('1')
        P3[race-1] = lookup[num] / nCr(6, num)
    P3 = P3.reshape(1,1,63,1)
    P4 = np.array([0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]).reshape(1,1,1,8)
    P = P1 * P2 * P3 * P4   

    data = prng.choice(P.size, N, True, P.flatten())
    x = np.bincount(data, minlength=P.size) 
    return x.astype(float)

def run_test(W, A, x, eps = np.sqrt(2), trials = 1, seed=456, engine='nnls'):
    prng = np.random.RandomState(seed)
    delta = A.sensitivity()
    A = A.sparse_matrix()
    ans = W.dot(x)

    errors = []
    for _ in range(trials):
        y = A.dot(x) + prng.laplace(loc=0, scale=delta / eps, size=A.shape[0])
        if engine == 'nnls':
            xest = inference.nnls(A, y)
        elif engine == 'wnnls':
            xest = inference.wnnls(W, A, y)
        est = W.dot(xest)
        errors.append( np.sum( (ans - est)**2 ) )
      
    rmse = np.sqrt(np.mean(errors) / W.shape[0])
    print(rmse)
    return rmse


if __name__ == '__main__':

    #x = synthetic_data(N=1000000)

    engine = 'wnnls'
    W = pl94_workload()
    A1 = opt_p_identity(W)
    A2 = marginal_strategy(W)
    A3 = manual_strategy()

    err1 = error.rootmse(W, A1)
    err2 = error.rootmse(W, A2)
    err3 = error.rootmse(W, A3)

    trials = 25
    Ns = [10**k for k in range(1, 8)]

    errs1, errs2, errs3 = [], [], []
    for N in Ns:
        x = synthetic_data(N)
        errs1.append( run_test(W, A1, x, trials=trials, engine=engine) )
        errs2.append( run_test(W, A2, x, trials=trials, engine=engine) )
        errs3.append( run_test(W, A3, x, trials=trials, engine=engine) )

    plt.plot(Ns, [err1]*len(Ns), 'b', label='KronPIdentity+LS')
    plt.plot(Ns, [err2]*len(Ns), 'r', label='Marginals+LS')
    plt.plot(Ns, [err3]*len(Ns), 'k', label='Manual+LS')

    plt.plot(Ns, errs1, 'bo', label='KronPIdentity+%s' % engine.upper())
    plt.plot(Ns, errs2, 'ro', label='Marginals+%s' % engine.upper())
    plt.plot(Ns, errs3, 'ko', label='Manual+%s' % engine.upper())

    plt.xlabel('Number of People')
    plt.ylabel('RMSE')
    plt.xscale('log')
    plt.legend()
    plt.savefig('pl94_%s.png' % engine)
    plt.show()
