import workload
from census_workloads import *
import optimize
import approximation
import implicit
import time
from IPython import embed


base = '/home/ryan/Desktop/privbayes_data/new/numpy'
eps = 1.0
X = np.load('%s/census.npy' % base)

for geography in [True]: #[False, True]:
    if geography:
        x = X.flatten()
    else:
        x = X.sum(axis=5).flatten()
    for compact in [False]:# [False, True]:
        if compact:
            sf1 = CensusSF1(geography)
        else:
            sf1 = CensusSF1Big(geography, reallybig=False)

        t0 = time.time()
        W = implicit.stack(*[implicit.krons(*[S.W for S in K.workloads]) for K in sf1.workloads])
        marg = approximation.marginals_approx(sf1)
        q = sf1.queries
        split = len(sf1.workloads) / 2
        sf1a = workload.Concat(sf1.workloads[:split])
        sf1b = workload.Concat(sf1.workloads[split:]) 

        strategy = optimize.restart_union_kron(sf1, 25, [1,1,8,1,10,1])
        theta, phi = optimize.restart_marginals(marg, 25)
        A1 = optimize.restart_union_kron(sf1a, 25, [1,1,8,1,10,1])
        A2 = optimize.restart_union_kron(sf1b, 25, [1,1,8,1,10,1])

        for A in strategy:
            A[A < 1e-3] = 0.0

        err1 = sf1.expected_error(strategy, eps)
        err2 = marg.expected_error(theta, eps)

        t1 = time.time()

    #    print np.sqrt(err1 / q), np.sqrt(err2 / q)

        A = implicit.krons(*strategy)
        A1 = implicit.krons(*[np.linalg.pinv(Ai) for Ai in strategy])
        y = A.dot(x) + np.random.laplace(0, 1.0/eps, A.shape[0])
        xhat = A1.dot(y)
        t2 = time.time()
        ans = W.dot(xhat)
        t3 = time.time()

        #err = sf1.squared_error(x - xhat)

        print geography, len(sf1.workloads), t1-t0, t2-t1, t3-t2, t3-t0
