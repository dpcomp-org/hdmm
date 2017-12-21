import numpy as np
from IPython import embed
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy import sparse
import time
import matplotlib.pyplot as plt
import optimize
import workload
import implicit
import pickle

def inverse(B):
    scale = 1.0 + B.sum(axis=0)
    m, n = B.shape
    C = np.linalg.inv(np.eye(m) + B.dot(B.T))
    At = sparse.vstack([sparse.diags(1.0/scale), B/scale]).T
    def matvec(v):
        if v.ndim == 2:
            return matmat(v)
        u = At.dot(v)
        u1 = scale**2 * u
        u2 = scale*B.T.dot(C.dot(B.dot(scale * u)))
        return u1 - u2
    def matmat(V):
        U = At.dot(V)
        U1 = scale[:,None]**2 * U
        U2 = scale[:,None]*B.T.dot(C.dot(B.dot(scale[:,None] * U)))
        return U1 - U2
    return LinearOperator(shape=(n, m+n), matvec=matvec, rmatvec=None, matmat=matmat)

def inverse2(B):
    scale = 1.0 + B.sum(axis=0)
    m, n = B.shape
    C = np.linalg.inv(np.eye(m) + B.dot(B.T))
    At = sparse.vstack([sparse.diags(1.0/scale), B/scale], dtype=np.float32).T
    BD = B*scale
    BTD0 = (B/scale).T
    CBD = C.dot(BD)
    def matvec(v):
        if v.ndim == 2:
            return matmat(v)
        u = At.dot(v)
        u1 = scale**2 * u
        u2 = BD.T.dot(C.dot(BD.dot(u)))
        return u1 - u2
    def matmat(V):
        U = V[:n]/scale[:,None] + BTD0.dot(V[n:])
#        U = np.copy(V[:n])
#        U = At.dot(V)
#        U2 = BD.T.dot(C.dot(BD.dot(U)))
#        U2 = np.linalg.multi_dot([BD.T, C, BD, U])
        U2 = BD.T.dot(CBD.dot(U))
        U *= scale[:,None]**2
        U -= U2
        return U
    return LinearOperator(shape=(n, m+n), matvec=matvec, rmatvec=None, matmat=matmat, dtype=np.float32)


def krons(*mats):
    mats = [aslinearoperator(A) for A in mats]
    N = np.prod([A.shape[1] for A in mats])
    M = np.prod([A.shape[0] for A in mats])
    def matvec(x):
        size = N
        X = x    
        for A in mats[::-1]:
            m, n = A.shape
            X = A * X.reshape(size//n, n).T
            size = size * m // n
        return X.flatten() 
    return LinearOperator(shape=(M, N), matvec=matvec, dtype=np.float32)

        

def run_mechanism(A, A1, eps=0.1):
    # assumes A has sensitivity 1
    x = np.random.geometric(0.2, A.shape[1]).astype(np.float32)
    t0 = time.time()
    y = A.dot(x) + np.random.laplace(loc=0.0, scale=1.0/eps, size=A.shape[0]).astype(np.float32)
#    print A.shape[1], time.time() - t0
    xhat = A1.dot(y)
    return time.time() - t0     

if __name__ == '__main__':
    
    strategies = {}
    inverses = {}


    for n in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        strategies[n] = np.load('/home/ryan/Desktop/strategies/all-range_%d_%d.npy' % (n, n//16)).astype(np.float32)
        B = strategies[n][n:] / np.diag(strategies[n][:n])
        inverses[n] = inverse2(B)


    # 3d range query strategy

    dom2d = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    dom3d = np.array([16, 32, 64, 128, 256, 512, 1024])
    dom4d = np.array([16, 32, 64, 128])

    dom2d = dom2d[1:]
    dom3d = dom3d[1:]
    dom4d = dom4d[1:]
    domdc = np.arange(2, 10)

    try:
        time2d, time3d, time4d, timesum, timedc = pickle.load(open('scalability_running.pkl', 'rb'))
        print 'checkpt try'
    except:

        time2d = []
        time3d = []
        time4d = []
        timesum = []
        timedc = []

        for n in dom2d:
            A = sparse.csr_matrix(strategies[n])
            T = np.ones((1,n))
            B1 = implicit.krons(A, T)
            B2 = implicit.krons(T, A)
            B = implicit.stack(B1, B2)
        
#            B1 = sparse.kron(A, T, format='csr')
#            B2 = sparse.kron(T, A, format='csr')
#            B = sparse.vstack([B1, B2], format='csr')

            timesum.append(run_mechanism(B, implicit.sparse_inverse(B)))
            print n, timesum[-1] 

    #    embed()
        # 2d domains
        for n in dom2d:
            print n
            A = strategies[n]
        #    A1 = inverses[n]
            A1 = np.linalg.pinv(A)
            time2d.append(run_mechanism(krons(A,A), krons(A1,A1)))
            if n in dom3d:
                time3d.append(run_mechanism(krons(A,A,A), krons(A1,A1,A1)))
            if n in dom4d:
                time4d.append(run_mechanism(krons(A,A,A,A), krons(A1,A1,A1,A1)))

        print time2d[-1]
        print time3d[-1]
        print time4d[-1]

        # import sys; sys.exit()

        for d in domdc:
            print d
            dom = tuple([10]*d)
            dc = workload.LowDimMarginals(dom, (d+1)//2)
            best_theta = None
            best_fun = np.inf
            print 'optimizing'
            for i in range(10):
                ans = optimize.optimize_marginals(dom, dc.weight_vector())
                if ans['error'] < best_fun:
                    best_fun = ans['error']
                    best_theta = ans['theta'].astype(np.float32)
                    best_inv = ans['invtheta'].astype(np.float32)
            print 'least squares'
            A = implicit.marginals_linop(dom, best_theta)
            A1 = implicit.marginals_inverse(dom, best_theta, best_inv)
            timedc.append(run_mechanism(A, A1))
            print A.shape, timedc[-1]
        
        obj = (time2d, time3d, time4d, timesum, timedc)
        pickle.dump(obj, open('scalability_running.pkl', 'wb'))

    plt.plot(dom2d**2, timesum, 'o-')
    plt.plot(dom2d**2, time2d, 'o-', label='2D Range Queries')
    plt.plot(dom3d**3, time3d, 'o-', label='3D Range Queries')
    plt.plot(dom4d**4, time4d, 'o-', label='4D Range Queries')
    plt.plot(10**domdc[1:], timedc[1:], 'o-', label='Marginals')
    plt.loglog()
    plt.legend(loc='lower right')
    plt.xlabel('Domain Size')
    plt.ylabel('Time (s)')
    plt.savefig('scalability_mechanism.png')
    plt.savefig('scalability_mechanism.pdf')
    plt.show()
