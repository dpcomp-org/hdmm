import numpy as np
from IPython import embed
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy import sparse
import time
import matplotlib.pyplot as plt
import itertools

def inverse(B):
    dtype = B.dtype
    scale = 1.0 + B.sum(axis=0)
    m, n = B.shape
    C = np.linalg.inv(np.eye(m) + B.dot(B.T))
    At = sparse.vstack([sparse.diags(1.0/scale), B/scale], dtype=dtype).T
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
        U2 = BD.T.dot(CBD.dot(U))
        U *= scale[:,None]**2
        U -= U2
        return U
    return LinearOperator(shape=(n, m+n), matvec=matvec, rmatvec=None, matmat=matmat, dtype=dtype)

def krons(*mats):
    dtype = mats[0].dtype
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
    def rmatvec(y):
        size = M
        Y =  y
        for A in mats[::-1]:
            m, n = A.shape
            Y = A.H * Y.reshape(size//m, m).T
            size = size * n // m
        return Y.flatten()
    return LinearOperator(shape=(M, N), matvec=matvec, rmatvec=rmatvec, dtype=dtype)

def stack(*mats):
    dtype = mats[0].dtype
    mats = [aslinearoperator(A) for A in mats]
    N = mats[0].shape[1]
    M = sum(A.shape[0] for A in mats)
    idx = np.cumsum([A.shape[0] for A in mats])[:-1]
    
    def matvec(x):
        return np.concatenate([A.dot(x) for A in mats]) 
    def rmatvec(y):
        ys = np.split(y, idx)
        return sum([A.rmatvec(z) for A, z in zip(mats, ys)])
    return LinearOperator(shape=(M,N), matvec=matvec, rmatvec=rmatvec, dtype=dtype)

def sparse_inverse(A):
    def matvec(y):
        return lsmr(A, y, atol=1e-9, btol=1e-9)[0]
    m, n = A.shape
    return LinearOperator(shape=(n, m), matvec=matvec, dtype=np.float32)

def _MtM_linop(domain, weights):
    dtype = weights.dtype
    d = len(domain)
    n = np.prod(domain)
    def axes(i):
        return tuple([k for k in range(d) if not i & 2**k])
    zipped = [(weights[i], axes(i)) for i in range(2**d) if weights[i] != 0]
    def matvec(v):
        X = v.reshape(domain)
        Y = np.zeros(domain, dtype=dtype)
        for w, ax in zipped:
            Y += w*X.sum(axis=ax, keepdims=True)
        return Y.flatten()
    return LinearOperator((n,n), matvec, matvec, dtype=dtype)

def marginals_linop(domain, weights):
    d = len(domain)
    def axes(i):
        return tuple([k for k in range(d) if not i & 2**k])
    def size(i):
        return np.prod([domain[k] for k in range(d) if i & 2**k], dtype=int)
    zipped = [(weights[i], axes(i)) for i in range(2**d) if weights[i] != 0]
    sizes = [size(i) for i in range(2**d) if weights[i] != 0]
    idx = np.cumsum(sizes)[:-1]
    def matvec(x):
        X = x.reshape(domain)
        return np.concatenate([w*X.sum(axis=ax).flatten() for w,ax in zipped])
    def rmatvec(y):
        ys = np.split(y, idx)
        X = np.zeros(domain)
        for (w,ax), z in zip(zipped, ys):
            tmp = tuple([1 if k in ax else domain[k] for k in range(d)])
            X += w*z.reshape(tmp)
        return X.flatten()
    
    m, n = sum(sizes), np.prod(domain)
    return LinearOperator((m,n), matvec, rmatvec, dtype=weights.dtype)

def marginals_inverse(domain, weights, invweights):
    return _MtM_linop(domain, invweights) * marginals_linop(domain, weights).H

