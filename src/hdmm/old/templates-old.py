import numpy as np
from scipy import sparse, optimize
from scipy.sparse.linalg import spsolve_triangular, aslinearoperator
from ektelo import workload
import time
import approximation
import implicit
from functools import reduce


class TemplateStrategy:
    """
    A template strategy is a space of strategies parameterized by some vector of weights.
    This weight vector can be optimized for a particular workload.
    """
    def __init__(self, theta0=None):
        self.set_params(theta0)
    
    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = np.maximum(theta, 0)

    def run_mechanism(self, x, eps):
        """ Run the matrix mechanism with the current strategy on the given data vector """
        A = self.strategy()
        A1 = self.inverse()
        delta = self.sensitivity()
        y = A.dot(x) + np.random.laplace(0, delta/eps, size=A.shape[0])
        return A1.dot(y)

    @property
    def A(self):
        return self.strategy(form='matrix')
    
    def _AtA1(self):
        return np.linalg.pinv(self.A.T.dot(self.A))

    def sparse_matrix(self):
        return sparse.csr_matrix(self.A)

    def strategy(self, form='linop'):
        """ Return a linear operator for the strategy """
        assert form in ['matrix', 'linop']
        A = self._strategy()
        if form == 'matrix':
            I = np.eye(A.shape[1])
            return A.dot(I)
        return A

    def _strategy(self):
        return aslinearoperator(self.A)

    def inverse(self, form='linop'):
        """ Return a linear operator for the pseudo inverse of the strategy """
        assert form in ['matrix', 'linop']
        A1 = self._inverse()
        if form == 'matrix':
            I = np.eye(A1.shape[1])
            return A1.dot(I)
        return A1

    def _inverse(self):
        return aslinearoperator(np.linalg.pinv(self.A))

    def _loss_and_grad(self):
        pass

    def sensitivity(self):
        return np.abs(self.A).sum(axis=0).max()

    def set_workload(self, W):
        self.workload = W

    def optimize(self, W):
        """
        Optimize strategy for given workload 

        :param W: the workload, may be a n x n numpy array for WtW or a workload object
        """
        t0 = time.time()
        self.set_workload(W)
        init = self.get_params()
        bnds = [(0,None)] * init.size
        log = []
        
        def obj(theta):
            self.set_params(theta)
            ans = self._loss_and_grad()
            log.append(ans[0])
            return ans

        opts = { 'ftol' : 1e-4 }
        res = optimize.minimize(obj, init, jac=True, method='L-BFGS-B', bounds=bnds, options=opts)
        t1 = time.time()
        params = self.get_params()
        ans = { 'log' : log, 'time' : t1 - t0, 'loss' : res.fun, 'res' : res, 'params' : params }
        return ans

    def restart_optimize(self, W, restarts):
        best = self.optimize(W)
        for i in range(restarts-1):
            ans = self.optimize(W)
            if ans['loss'] < best['loss']:
                best = ans
        self.set_params(best['params'])
        return best 

class Default(TemplateStrategy):
    """
    """
    def __init__(self, m, n):
        theta0 = np.random.rand(m*n)
        self.m = m
        self.n = n
        TemplateStrategy.__init__(theta0) 

    def _strategy(self):
        return self.get_params().reshape(self.m, self.n)

    def _loss_and_grad(self):
        WtW = self.workload.WtW
        A = self.get_params().reshape(self.m, self.n)
        sums = np.sum(np.abs(A), axis=0)
        col = np.argmax(sums)
        F = sums[col]**2
        # note: F is not differentiable, but we can take subgradients
        dF = np.zeros_like(A)
        dF[:,col] = np.sign(A[:,col])*2*sums[col]
        AtA = A.T.dot(A)
        AtA1 = np.linalg.pinv(AtA)
        M = WtW.dot(AtA1)
        G = np.trace(M)
        dX = -AtA1.dot(M)
        dG = 2*A.dot(dX)
        dA = dF*G + F*dG
        return F*G, dA.flatten()

class PIdentity(TemplateStrategy):
    """
    A PIdentity strategy is a strategy of the form (I + B) D where D is a diagonal scaling matrix
    that depends on B and ensures uniform column norm.  B is a p x n matrix of free parameters.
    """
    def __init__(self, p, n):
        """
        Initialize a PIdentity strategy
        :param p: the number of non-identity queries
        :param n: the domain size
        """
        theta0 = np.random.rand(p*n)
        self.p = p
        self.n = n
        TemplateStrategy.__init__(self, theta0)
   
    def sparse_matrix(self):
        I = sparse.identity(self.n, format='csr')
        B = self.get_params().reshape(self.p, self.n)
        D = 1 + B.sum(axis=0)
        A = sparse.vstack([I,B], format='csr')
        return A * sparse.diags(1.0 / D)
 
    def _strategy(self):
        I = np.eye(self.n)
        B = self.get_params().reshape(self.p, self.n)
        A = np.vstack([I, B])
        A = A / A.sum(axis=0)
        return aslinearoperator(sparse.csr_matrix(A))

    def _AtA1(self):
        B = self.get_params().reshape(self.p, self.n)
        R = np.linalg.inv(np.eye(self.p) + B.dot(B.T))
        D = 1.0 + B.sum(axis=0)
        return (np.eye(self.n) - B.T.dot(R).dot(B))*D*D[:,None]

    def _inverse(self):
        B = self.get_params().reshape(self.p, self.n)
        return implicit.inverse(B)
        
    def _loss_and_grad(self):
        WtW = self.workload.WtW
        p, n = self.p, self.n

        B = np.reshape(self.get_params(), (p,n))
        scale = 1.0 + np.sum(B, axis=0)
        R = np.linalg.inv(np.eye(p) + B.dot(B.T)) # O(k^3)
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
        return loss, grad.flatten()

class AugmentedIdentity(TemplateStrategy):
    """
    An AugmentedIdentity strategy is like a PIdentity strategy with additional structure imposed.
    The template is defiend by a p x n matrix of non-negative integers P.  Each unique nonzero entry
    of this matrix P refers to a free parameter that can be optimized.  An entry that is 0 in P is
    a structural zero in the strategy.  

    Example 1:
    A PIdentity strategy can be represented as an AugmentedIdentity strategy with 
    P = np.arange(1, p*n+1).reshape(p, n)
    
    Example 2:
    A strategy of the form w*T + I can be represented as an AugmentedIdentity strategy with
    P = np.ones((1, n), dtype=int)
    """
    def __init__(self, imatrix):
        """ 
        Create an AugmentedIdentity strategy with the given P matrix
        """
        self.imatrix = imatrix
        p, n = imatrix.shape
        num = imatrix.max()
        theta0 = np.random.rand(num)
        self._pid = PIdentity(p, n)
        TemplateStrategy.__init__(self, p+n, n, theta0)
        # should call set_params
     
    def _strategy(self):
        return self._pid._strategy()   
 
    def _inverse(self):
        return self._pid.inverse()

    def set_params(theta):
        self.theta = theta
        params = np.append(0, theta)
        B = params[self.imatrix]
        self._pid.set_params(B.flatten())

    def _AtA1(self):
        return self._pid._AtA1()

    def set_workload(self, W):
        self.workload = W
        self._pid.set_workload(W)
 
    def _loss_and_grad(self):
        #params = np.append(0, self.get_params())
        #B = params[self.imatrix]
        #self._pid.set_params(B.flatten())
        obj, grad = self._pid._loss_and_grad() 
        grad2 = np.bincount(self.imatrix.flatten(), grad)[1:]
        return obj, grad2

class Static(TemplateStrategy):
    def __init__(self, strategy):
        self.A = strategy
        TemplateStrategy.__init__(self, np.array([]))

    def optimize(self, W):
        pass

class Kronecker(TemplateStrategy):
    """ A Kronecker template strategy is of the form A1 x ... x Ad, where each Ai is some 1D 
        template strategy"""
    def __init__(self, strategies):
        """
        :param strategies: a list of templates for each dimension of template
        """
        self.strategies = strategies

    def sparse_matrix(self):
        return reduce(sparse.kron, [A.sparse_matrix() for A in self.strategies])

    def set_params(self, params):
        for strategy, param in zip(self.strategies, params):
            strategy.set_params(param)

    def get_params(self):
        return [strategy.get_params() for strategy in self.strategies]

    def _strategy(self):
        return implicit.krons(*[S._strategy() for S in self.strategies])

    def _inverse(self):
        return implicit.krons(*[S._inverse() for S in self.strategies])

    def sensitivity(self):
        return np.prod([S.sensitivity() for S in self.strategies])

    def optimize(self, W):
        self.set_workload(W)
        t0 = time.time()
        if isinstance(W, workload.Kron):
            loss = 0
            for subA, subW in zip(self.strategies, W.workloads):
                ans = subA.optimize(subW)
                loss += ans['loss']
            params = self.get_params()
            return { 'time' : time.time() - t0, 'loss' : loss, 'params' : params }
        assert isinstance(W, workload.Concat) and isinstance(W.workloads[0], workload.Kron)
      
        workloads = [K.workloads for K in W.workloads] # a k x d table of workloads
        strategies = self.strategies
 
        k = len(workloads)
        d = len(workloads[0])

        log = []

        C = np.ones((d, k))
        for i in range(d):
            AtA1 = strategies[i]._AtA1()
            for j in range(k):
                C[i,j] = np.sum(workloads[j][i].WtW * AtA1)
        for r in range(10):
            err = C.prod(axis=0).sum()
            for i in range(d):
                cs = np.sqrt(C.prod(axis=0) / C[i])
                What = workload.Concat([c*Ws[i] for c, Ws in zip(cs, workloads)])
                res = strategies[i].optimize(What)
                AtA1 = strategies[i]._AtA1()
                for j in range(k):
                    C[i,j] = np.sum(workloads[j][i].WtW * AtA1)
            log.append(err)

        t1 = time.time()
        params = self.get_params()
        ans = { 'log' : log, 'loss' : err, 'time' : t1 - t0, 'params' : params }
        return ans

class Marginals(TemplateStrategy):
    """
    A marginals template is parameterized by 2^d weights where d is the number of dimensions.  
    The strategy is of the form w_1 (T x ... x T) + ... + w_{2^d} (I x ... I)  - every marginal
    with nonzero weight is queried with weight w_i
    """
    def __init__(self, domain):
        self.domain = domain
        theta = np.random.rand(2**len(domain))

        d = len(domain)
        mult = np.ones(2**d)
        for i in range(2**d):
            for k in range(d):
                if not (i & (2**k)):
                    mult[i] *= domain[k]
        self.mult = mult

        TemplateStrategy.__init__(self, theta)

    def _strategy(self):
        return implicit.marginals_linop(self.domain, self.get_params())

    def _inverse(self):
        theta = self.get_params()
        Y, _ = self._Xmatrix(theta**2)
        tmp = Y.dot(theta**2)
        X, _ = self._Xmatrix(tmp)
        invtheta = spsolve_triangular(X, theta**2, lower=False)
        return implicit.marginals_inverse(self.domain, theta, invtheta)

    def sensitivity(self):
        return np.sum(np.abs(self.get_params()))

    def _Xmatrix(self,vect):
        # the matrix X such that M(u) M(v) = M(X(u) v)
        d = len(self.domain)
        A = np.arange(2**d)
        mult = self.mult

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
        return X, XT

    def set_workload(self, W):
        marg = approximation.marginals_approx(W)
        self.workload = marg
        d = len(self.domain)
        A = np.arange(2**d)
        weights = marg.weight_vector()
        self.dphi = np.array([np.dot(weights**2, self.mult[A|b]) for b in range(2**d)]) 

    def _loss_and_grad(self):
        d = len(self.domain)
        A = np.arange(2**d)
        mult = self.mult
        dphi = self.dphi
        theta = self.get_params()

        delta = np.sum(theta)**2
        ddelta = 2*np.sum(theta)
        theta2 = theta**2
        Y, YT = self._Xmatrix(theta2)
        params = Y.dot(theta2)
        X, XT = self._Xmatrix(params)
        phi = spsolve_triangular(X, theta2, lower=False)
        # Note: we should be multiplying by domain size here if we want total squared error
        ans = np.dot(phi, dphi)
        dXvect = -spsolve_triangular(XT, dphi, lower=True)
        # dX = outer(dXvect, phi)
        dparams = np.array([np.dot(dXvect[A&b]*phi, mult[A|b]) for b in range(2**d)])
        dtheta2 = YT.dot(dparams)
        dtheta = 2*theta*dtheta2
        return delta*ans, delta*dtheta + ddelta*ans

# (df / dtheta_k) = sum_ij (df / d_Aij) (dA_ij / theta_k)
  
def KronPIdentity(ns, ps):
    """
    Builds a template strategy of the form A1 x ... x Ad where each Ai is a PIdentity template
    :param ns: the domain size of each dimension
    :param ps: the number of p queries in each dimension
    """
    return Kronecker([PIdentity(p, n) for p,n in zip(ps, ns)])
 
def RangeTemplate(n, start=32, branch=4, shared=False):
    """
    Builds a template strategy for range queries with queries that have structural zeros 
    everywhere except at indices at [i, i+w) where w is the width of the query and ranges from
    start to n in powers of branch and i is a multiple of w/2.

    :param n: the domain size
    :param start: the width of the smallest query
    :param branch: the width multiplying factor for larger queries
    :param shared: flag to determine if parameters should be shared for queries of the same width

    Example:
    RangeTemplate(16, start=8, branch=2) builds a strategy template with four augmented queries that have structural zeros everywhere except in the intervals indicated below:
    1. [0,8)
    2. [4,12)
    3. [8,16)
    4. [0,16)
    """
    rows = []
    width = start
    idx = 1
    while width <= n:
        for i in range(0, n-width//2, width//2):
            row = np.zeros(n, dtype=int)
            row[i:i+width] = np.arange(width) + idx
            if not shared: idx += width
            rows.append(row)
        if shared: idx += width
        width *= branch
    return AugmentedIdentity(np.vstack(rows))

def IdTotal(n):
    """ Build a single-parameter template strategy of the form w*Total + Identity """
    P = np.ones((1,n), dtype=int)
    return AugmentedIdentity(P)

def Identity(n):
    """ Builds a template strategy that is always Identity """
    return Static(np.eye(n))

def Total(n):
    """ Builds a template strategy that is always Total """
    return Static(np.ones((1,n)))

