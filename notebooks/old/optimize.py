import numpy as np
from scipy import sparse
import scipy.linalg as spla
from scipy import optimize
from scipy.sparse.linalg import spsolve_triangular, lsmr
from scipy import sparse
from IPython import embed
import itertools
from workload import *
import time

def direct_optimization(WtW, A0):
    k, n = A0.shape
    
    def loss_and_grad(b):
        B = b.reshape(k, n)
        scale = B.sum(axis=0)
        A = B / scale
        X1 = spla.pinvh(A.T.dot(A))
        M = WtW.dot(X1)
        dfX = -X1.dot(M)
        dfA = 2*A.dot(dfX)
        dfB = (dfA*scale - (B*dfA).sum(axis=0)) / scale**2
        return np.trace(M), dfB.flatten()

    bnds = [(0,None)]*(k*n)
    
    res = optimize.minimize(loss_and_grad, x0=A0.flatten(), jac=True, method='L-BFGS-B', bounds=bnds)
    A = res.x.reshape(k, n) 
    A /= A.sum(axis=0)
    return A, res

def restart_optimize(W, restarts, p):
    """ accepts a workload object """ 
    WtW = W.WtW
    best = np.inf
    n = WtW.shape[0]
    for i in range(restarts):
        ans = augmented_optimization(WtW, np.random.rand(p, n))
        if ans['res'].fun < best:
            best = ans['res'].fun
            result = ans
    return result['A']

def restart_union_kron(W, restarts, ps):
    """ accepts a workload object """
    WtWs = [[w2.WtW for w2 in w1.workloads] for w1 in W.workloads]
    best = np.inf
    ns = W.domain
    for i in range(restarts):
        init = [np.random.rand(p, n) for p, n in zip(ps, ns)]
        ans = union_kron(WtWs, init)
        if ans['error'] < best:
            best = ans['error']
            result = ans
    return result['As']

def restart_marginals(W, restarts):
    weights = W.weight_vector()
    dom = W.domain
    ans = optimize_marginals(dom, weights, weights)
    best = ans
    for i in range(restarts):
        ans = optimize_marginals(dom, weights)
        if ans['valid'] and ans['error'] < best['error']:
            best = ans
    return best['theta'], best['invtheta']  

def restart_kron(W, restarts, ps):
    As = [None]*len(ps)
    for i in range(len(ps)):
        As[i] = restart_optimize(W.workloads[i], restarts, ps[i])
    return As
 

def augmented_optimization(WtW, B0):
    """ 
    """
    n = WtW.shape[0]
    if B0 is None:
        B0 = np.random.rand(n//16, n)
    k, n = B0.shape
    log = []

    def loss_and_grad(b):
        B = np.reshape(b, (k,n))
        scale = 1.0 + np.sum(B, axis=0)
        R = np.linalg.inv(np.eye(k) + B.dot(B.T)) # O(k^3)
        C = WtW * scale * scale[:,None] # O(n^2)

        M1 = R.dot(B) # O(n k^2)
        M2 = M1.dot(C) # O(n^2 k)
        M3 = B.T.dot(M2) # O(n^2 k)
        M4 = B.T.dot(M2.dot(M1.T)).dot(B) # O(n^2 k)

        Z = -(C - M3 - M3.T + M4) * scale * scale[:,None] # O(n^2)

        Y1 = 2*np.diag(Z) / scale # O(n)
        Y2 = 2*(B/scale).dot(Z) # O(n^2 k)
        g = Y1 + (B*Y2).sum(axis=0) # O(n k)

        loss = np.trace(C) - np.trace(M3)
        grad = (Y2*scale - g) / scale**2
        log.append(loss)
        return loss, grad.flatten()
    
    bnds = [(0,None)]*(k*n)

    opts = { 'ftol' : 1e-4 }
    t0 = time.time()
    res=optimize.minimize(loss_and_grad,x0=B0.flatten(),jac=True,method='L-BFGS-B',bounds=bnds,options=opts)
    t1 = time.time()
    B = res.x.reshape(k,n)
    A = np.vstack([np.eye(n), B])
    ans = {}
    ans['A'] = A / A.sum(axis=0)
    ans['B'] = B
    ans['error'] = res.fun
    ans['res'] = res
    ans['log'] = np.array(log)
    ans['time'] = t1 - t0
    return ans

def union_kron(workloads, init, cycles = 10):
    """
    :param workloads: m x d table of workloads in normal form where W^i = W^i_1 x ... W^i_d for 1<=i<=m
    """
    def inverse(B):
        p, n = B.shape
        R = np.linalg.inv(np.eye(p) + B.dot(B.T))
        D = 1.0 + B.sum(axis=0)
        return (np.eye(n) - B.T.dot(R).dot(B))*D*D[:,None]  
 
    k = len(workloads)
    d = len(workloads[0])
    
    Bs = init
    As = [None]*d
    
    t0 = time.time()
    log = []

    C = np.ones((d, k))
    for i in range(d):
        AtA1 = inverse(Bs[i])
        for j in range(k):
            C[i,j] = np.sum(workloads[j][i] * AtA1)
    for r in range(cycles):
        err = C.prod(axis=0).sum()
        for i in range(d):
            cs = C.prod(axis=0) / C[i]
            WtW = sum(c*WtWs[i] for c, WtWs in zip(cs, workloads))
#            B0 = np.random.rand(*Bs[i].shape)
            ans = augmented_optimization(WtW, Bs[i])
            As[i] = ans['A']
            Bs[i] = ans['B']
            AtA1 = inverse(Bs[i])
            for j in range(k):
                C[i,j] = np.sum(workloads[j][i] * AtA1)
        log.append(err)

    t1 = time.time()
    ans = { 'As' : As, 'log' : log, 'error' : err, 'time' : t1 - t0 }
    return ans

def optimize_marginals(dom, weights, init=None):
    d = len(dom)
    mult = np.ones(2**d)
    for i in range(2**d):
        for k in range(d):
            if not (i & (2**k)):
                mult[i] *= dom[k]
    A = np.arange(2**d)
    log = []

    def Xmatrix(vect):
        values = np.zeros(3**d)
        rows = np.zeros(3**d, dtype=int)
        cols = np.zeros(3**d, dtype=int)
        start = 0
        for b in range(2**d):
            #uniq, rev = np.unique(a&B, return_inverse=True) # most of time being spent here
            mask = np.zeros(2**d, dtype=int)
            mask[A&b] = 1
            uniq = np.nonzero(mask)[0]
            step = uniq.size
            mask[uniq] = np.arange(step)
            rev = mask[A&b]
            values[start:start+step] = np.bincount(rev, vect*mult[A|b], step)
            if values[start+step-1] == 0:
                values[start+step-1] = 1.0 # hack to make solve triangular work
            cols[start:start+step] = b
            rows[start:start+step] = uniq
            start += step
        X = sparse.csr_matrix((values, (rows, cols)), (2**d, 2**d))
        XT = sparse.csr_matrix((values, (cols, rows)), (2**d, 2**d))
        # Note: If X is not full rank, need to modify it so that solve_triangular works
        # This doesn't impact the gradient calculations though
        # a finite difference sanity check might suggest otherwise, 
        # but a valid subgradient at theta_k = 0 is 0 due to symmetry
        #D = sparse.diags((X.diagonal()==0).astype(np.float64), format='csr')
        return X, XT

    dphi = np.array([np.dot(weights**2, mult[A|b]) for b in range(2**d)])
    #D = sparse.diags(np.ones(2**d) * h, format='csr', dtype=np.float64)
    def loss_and_grad(theta):
        theta[theta<1e-7] = 0.0
        delta = np.sum(theta)**2
        ddelta = 2*np.sum(theta)
        theta2 = theta**2
        Y, YT = Xmatrix(theta2)
        params = Y.dot(theta2)
        X, XT = Xmatrix(params)
        phi = spsolve_triangular(X, theta2, lower=False)
#        phi = lsmr(X, theta2, damp=1.0, atol=0, btol=0)
        # Note: we should be multiplying by domain size here if we want total squared error
        ans = np.dot(phi, dphi)
#        ans = np.sum([phi[b]*np.dot(mult[A|b],weights**2) for b in range(2**d)])
#        dphi = np.array([np.dot(weights**2, mult[A|b]) for b in range(2**d)])
        dXvect = -spsolve_triangular(XT, dphi, lower=True)
        # dX = outer(dXvect, phi)
        dparams = np.array([np.dot(dXvect[A&b]*phi, mult[A|b]) for b in range(2**d)])
        dtheta2 = YT.dot(dparams)
        dtheta = 2*theta*dtheta2
        if len(log) == 0 or delta*ans < log[-1]:
            log.append(delta*ans)
        #print delta*ans, np.log10(X.max())
        return delta*ans, delta*dtheta + ddelta*ans

    bnds = [(0,None)] * 2**d
    opts = { 'gtol' : 0 } #{ 'ftol' : 1e-4 }

    eye = np.zeros(2**d)
    eye[-1] = 1.0
    eye, _ = loss_and_grad(eye)
    workload, _= loss_and_grad(weights)

    if init is None:
        init = np.random.rand(weights.size)
        init /= init.sum()
        coef = np.zeros(2**d)
        for i in range(2**d):
            coef[i] = bin(i).count('1')
#        init = np.random.exponential(2**coef)
   
    t0 = time.time()
    res = optimize.minimize(loss_and_grad,init,method='L-BFGS-B',bounds=bnds,jac=True,options=opts)
    t1 = time.time()

    def recover_inverse(theta):
        theta2 = theta**2
        Y, _ = Xmatrix(theta2)
        params = Y.dot(theta2)
        X, _ = Xmatrix(params)
        return spsolve_triangular(X, theta2, lower=False)

    def check_valid(theta, phi):
        vect = Xmatrix(phi)[0].dot(theta**2)
        M = Xmatrix(vect)[0]
        diff = M.dot(weights) - weights
        return np.allclose(diff, 0)

    theta = np.maximum(res.x, 0)
    theta = theta / theta.sum()
    theta[theta<1e-10] = 0.0
    phi = recover_inverse(theta)

    ans = {}
    ans['init'] = init
    ans['error'] = res.fun * np.prod(dom)
    ans['theta'] = theta
    ans['invtheta'] = phi
    ans['identity'] = eye * np.prod(dom)
    ans['workload'] = workload * np.prod(dom)
    ans['res'] = res
    ans['log'] = np.array(log[2:]) * np.prod(dom)
    ans['time'] = t1 - t0
    ans['valid'] = check_valid(theta, phi)

    return ans

if __name__ == '__main__':
    workloads = [[Prefix(64).WtW, Identity(32).WtW],
                 [AllRange(64).WtW, Total(32).WtW],
                 [(100*WidthKRange(64, [3])).WtW, Total(32).WtW],
                 [Identity(64).WtW, Identity(32).WtW]]
    init = [np.random.rand(12, 64), np.random.rand(4, 32)]
    As = union_kron(workloads, init)
