import optimize
import workload
import itertools
import numpy as np
from IPython import embed
import utility
import pickle
from datacube import datacube_strategy

def fourier(alphas):
    d = len(alphas[0])
    betas = set()
    for beta in itertools.product(*[[0,1]]*d):
        for alpha in alphas:
            if all(a==b or b==0 for a, b in zip(alpha, beta)):
                betas.add(beta)
                continue
    A = np.zeros((len(betas), 2**d))
    for i, beta in enumerate(sorted(list(betas))):
        for j, alpha in enumerate(itertools.product(*[[0,1]]*d)):
            A[i,j] = (-1)**np.dot(alpha, beta)
    A /= np.abs(A).sum(axis=0)
    return A

def binary():
    d = 8
    dom = tuple([2]*d)
    cubes = list(itertools.product(*[[0,1]]*d))
    restarts = 50

    print 'Domain:', dom
    print 'k, GlobalOpt, Identity, Workload, Fourier'

    for k in range(d+1): 
        weights = {}
        for c in cubes:
            if sum(c) == k:
                weights[c] = 1.0
        fourierA = fourier(weights.keys())
        kway = Marginals(dom, weights)
        fourierErr = utility.rootmse(kway.WtW, fourierA, kway.queries)
        assert utility.supports(fourierA, kway.WtW)
        best = np.inf
        for i in range(restarts):
            res, errors = datacubes_optimization(kway)
    #        print k, i, errors
            if errors[0] < best:
                best = errors[0]
        print '%d, %.2f, %.2f, %.2f, %.2f' % (k, best, errors[1], errors[2], fourierErr)

def compute():
    d = 8
    dom = tuple([10]*d)
    restarts = 25

    print 'Domain:', dom
    print 'k, Identity, Workload, OPT_M, OPT_K, OPT_+, DataCube, Laplace'

    for k in range(1,d+1):
#        kway = workload.DimKMarginals(dom, [k]) 
        kway = workload.DimKMarginals(dom, range(k+1))
        split = len(kway.workloads) / 2
        kway1 = workload.Concat(kway.workloads[:split])
        kway2 = workload.Concat(kway.workloads[split:])

        ding = datacube_strategy(kway)
        theta, phi = optimize.restart_marginals(kway, restarts)
        A = optimize.restart_union_kron(kway, restarts, [1]*d)
        A1 = optimize.restart_union_kron(kway1, restarts, [1]*d)
        A2 = optimize.restart_union_kron(kway2, restarts, [1]*d)
        weights = kway.weight_vector()
        eye = np.zeros(2**d); eye[-1] = 1.0
        err1 = kway.expected_error(eye)
        err2 = kway.expected_error(weights)
        err3 = kway.expected_error(theta)
        err4 = workload.Concat.expected_error(kway, A)
        err5 = 4*workload.Concat.expected_error(kway1, A1) + 4*workload.Concat.expected_error(kway2, A2)
        err6 = kway.expected_error(ding)
        err7 = weights.sum()**2 * kway.queries

        pmm = min(err3, err1)

        print '%d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (k, np.sqrt(err1/pmm), np.sqrt(err2/pmm), np.sqrt(pmm/pmm), np.sqrt(err4/pmm), np.sqrt(err5/pmm), np.sqrt(err6/pmm), np.sqrt(err7/pmm))

def preloaded():
    base = '/home/ryan/Desktop/strategies'
    workload = 'lowd'
    n = 16
    d = 8
    print 'k, GlobalOpt, Identity, Workload'
    for k in range(d+1):
        best = None
        for trial in range(25):
            ans = pickle.load(open('%s/%s-%d-%d-%d-%d.pkl' % (base, workload, n, d, k, trial), 'rb'))
            if ans['valid'] and (best is None or ans['error'] < best['error']):
                best = ans
        pickle.dump(best, open('%s/%s-%d-%d-%d-best.pkl' % (base, workload, n, d, k), 'wb'))
        eye = ans['identity']
        work = ans['workload']
        best = np.inf if best is None else best['error']
        best = min(best, eye, work)
        print '%d, %.2f, %.2f, %.2f' % (k, 1.0, np.sqrt(eye/best), np.sqrt(work/best))

if __name__ == '__main__':
    compute()
