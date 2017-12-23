import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, aslinearoperator
import workload

def lstsq(A, y):
    res = lsqr(A, y)
    return res[0]

def loss_and_grad(P, WtW, y):
    """ 
    P: a un-normalized strategy matrix (may be represented as a LinearOperator)
    WtW: a workload matrix (may be represented as a LinearOperator)
    y: a noise vector

    return the objective and gradient in outer product form 
    """
    p, n = P.shape
    P = aslinearoperator(P)
    
    scale = P.H.dot(np.ones(p))
    D = aslinearoperator(sparse.diags(1.0/scale))
    
    A = P * D
    At = A.H

    z = lstsq(A, y)
    dz = 2 * WtW.dot(z)
    ans = z.dot(dz) / 2.0
    
    # term1 = -outer(z, a)
    a = lstsq(At, dz)
    #term2 = outer(b, c)
    b = lstsq(A, a)
    c = y - lstsq(At, At.dot(y))
    # term3 = outer(d, e)
    d = dz - lstsq(A, A.dot(dz))
    e = lstsq(At, z)

    U = np.vstack([-a,c,e])
    V = np.vstack([z,b,d])

    u0 = -np.ones(p)
    v0 = np.sum(V * P.H.dot(U.T).T, axis=0) / scale**2
    
    return ans, (np.vstack([u0, U]), np.vstack([v0, V/scale]))

def stochastic_p_identity(W, p):
    WtW = W.WtW
    n = WtW.shape[0]
    I = np.eye(n)

    a = 0.02
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    
    B = np.zeros((p,n))
#    B = np.random.rand(p,n)
    m = np.zeros_like(B)
    v = np.zeros_like(B)

    for t in range(1, 1000):
        A = np.vstack([I, B])

        obj = 0
        grad = np.zeros_like(B)
        rep = 5
        for k in range(rep):
            y = np.random.laplace(loc=0, scale=1.0/np.sqrt(2), size=p+n)
            f, (U,V) = loss_and_grad(A, WtW, y)
            grad += U.T[n:].dot(V) / rep
            obj += f / rep

        strategy = A / A.sum(axis=0)
        err = W.expected_error(strategy)
        print obj, err, obj/err
#        print np.diag(strategy)
    
#        grad = U.T[n:].dot(V)
        m = b1 * m + (1-b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        B = B - a * mhat / (np.sqrt(vhat) + eps)
        B[B < 0] = 0
    A = np.vstack([I, B])
    return A / A.sum(axis=0)

def stochastic_marginal(W):
    WtW = W.WtW
    n = WtW.shape[0]
    I = np.eye(n)
    T = np.ones(n)

    a = 0.02
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    
    theta = np.ones(n)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    for t in range(1, 1000):
        A = np.vstack([theta*T, I])

        obj = 0
        grad = 0.0
        rep = 5
        for k in range(rep):
            y = np.random.laplace(loc=0, scale=1.0/np.sqrt(2), size=1+n)
            f, (U,V) = loss_and_grad(A, WtW, y)
            grad += U[:,0].dot(V) / rep
            obj += f / rep

        err = W.expected_error(A)
        print obj, err, obj/err
#        print np.diag(strategy)
    
#        grad = U.T[n:].dot(V)
        m = b1 * m + (1-b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        theta = theta - a * mhat / (np.sqrt(vhat) + eps)
        theta = np.maximum(theta, 0.0)
    return A


if __name__ == '__main__':
    P = np.vstack([np.eye(16), np.random.rand(2, 16)])
    W = np.random.rand(16, 16)
    WtW = W.T.dot(W)
    y = np.random.rand(P.shape[0])
    
    obj, (U,V) = loss_and_grad(P, WtW, y)
    
    grad = U.T.dot(V)
    print np.diag(grad)
   
    approx = np.zeros(16)
    for i in range(16):
        eps = 1e-5
        P[i,i] += eps
        obj1, _ = loss_and_grad(P, WtW, y)
        P[i,i] -= eps
        approx[i] = (obj1 - obj) / eps
    print approx
 

