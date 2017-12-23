import numpy as np
from scipy import sparse optimize
from scipy.sparse.linalg import spsolve_triangular
import workload
import time

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

    @property
    def A(self):
        pass
    
    def _AtA1(self):
        return np.linalg.pinv(self.A.T.dot(self.A))

    def inverse(self):
        return np.linalg.pinv(self.A)

    def loss_and_grad(self, WtW):
        pass

    def sensitivity(self):
        return np.abs(self.A).sum(axis=0).max()

    def optimize(self, W):
        """
        Optimize strategy for given workload 

        :param W: the workload, may be a n x n numpy array for WtW or a workload object
        """
        if isinstance(W, workload.Workload):
            WtW = W.WtW
        else:
            WtW = W
        init = self.get_params()
        bnds = [(0,None)] * init.size
        
        def obj(theta):
            self.set_params(theta)
            return self.loss_and_grad(WtW)

        res = optimize.minimize(obj, init, jac=True, method='L-BFGS-B', bounds=bnds)
        return res

    def restart_optimize(self, workload):
        pass

class Default(TemplateStrategy):
    """
    The Default template strategy is characterized by m x n matrix A' of free parameters
    where m is the number of queries.  The strategy is A' D where D is a diagonal scaling matrix 
    that ensures uniform column norm.
    """
    def __init__(self, m, n):
        theta0 = np.random.rand(m*n)
        self.m = m
        self.n = n
        TemplateStrategy.__init__(theta0) 

    @property
    def A(self):
        B = self.get_params().reshape(self.m, self.n)
        return B / B.sum(axis=0)

    def loss_and_grad(self, WtW):
        B = self.get_params().reshape(self.m, self.n)
        scale = B.sum(axis=0)
        A = B / scale
        AtA = A.T.dot(A)
        AtA1 = np.linalg.pinv(AtA)
        M = WtW.dot(AtA1)
        dX = -AtA1.dot(M)
        dA = 2*A.dot(dX)
        dB = (dfA*scale - (B*dfA).sum(axis=0)) / scale**2
        return np.trace(M), dB.flatten()

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
    
    @property
    def A(self):
        I = np.eye(self.n)
        B = self.get_params().reshape(self.p, self.n)
        A = np.vstack([I, B])
        return A / A.sum(axis=0)

    def _AtA1(self):
        B = self.get_params().reshape(self.p, self.n)
        R = np.linalg.inv(np.eye(self.p) + B.dot(B.T))
        D = 1.0 + B.sum(axis=0)
        return (np.eye(self.n) - B.T.dot(R).dot(B))*D*D[:,None]

    def inverse(self):
        return self._AtA1().dot(self.A.T)
        
    def loss_and_grad(self, WtW):
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
     
    @property 
    def A(self):
        params = np.append(0, self.get_params())
        B = params[self.imatrix]
        self._pid.set_params(B.flatten())
        return self._pid.A

    def set_params(theta):
        self.theta = theta
        params = np.append(0, theta)
        B = params[self.imatrix]
        self._pid.set_params(B.flatten())

    def _AtA1(self):
        return self._pid._AtA1()

    def inverse(self):
        return self._pid.inverse()
 
    def loss_and_grad(self, WtW):
        #params = np.append(0, self.get_params())
        #B = params[self.imatrix]
        #self._pid.set_params(B.flatten())
        obj, grad = self._pid.loss_and_grad(WtW) 
        grad2 = np.bincount(self.imatrix.flatten(), grad)[1:]
        return obj, grad2

class Kronecker(TemplateStrategy):
    """ A Kronecker template strategy is of the form A1 x ... x Ad, where each Ai is some 1D 
        template strategy"""
    def __init__(self, strategies):
        """
        :param strategies: a list of templates for each dimension of template
        """
        self.strategies = strategies

    def optimize(self, W):
        t0 = time.time()
        if isinstance(W, workload.Kron):
            for subA, subW in zip(self.strategies, W.workloads):
                subA.optimize(subW)
            return { 'time' : time.time() - t0 }
        assert isinstance(W, workload.Concat) and isinstance(W.workloads[0], workload.Kron)
      
        # TODO: use workload object instead of WtW so marginals template can be used
        # on subworkloads 
        workloads = [[S.WtW for S in K.workloads] for K in W.workloads]
        strategies = self.strategies
 
        k = len(workloads)
        d = len(workloads[0])

        log = []

        C = np.ones((d, k))
        for i in range(d):
            AtA1 = strategies[i]._AtA1()
            for j in range(k):
                C[i,j] = np.sum(workloads[j][i] * AtA1)
        for r in range(10):
            err = C.prod(axis=0).sum()
            for i in range(d):
                cs = C.prod(axis=0) / C[i]
                WtW = sum(c*WtWs[i] for c, WtWs in zip(cs, workloads))
                res = strategies[i].optimize(WtW)
                AtA1 = strategies[i]._AtA1()
                for j in range(k):
                    C[i,j] = np.sum(workloads[j][i] * AtA1)
            log.append(err)

        t1 = time.time()
        ans = { 'log' : log, 'error' : err, 'time' : t1 - t0 }
        return ans

class Marginals(TemplateStrategy):
    def __init__(self, domain):
        self.domain = domain
        theta = np.random.rand(2**len(domain))

        d = len(domain)
        mult = np.ones(2**d)
        for i in range(2**d):
            for k in range(d):
                if not (i & (2**k)):
                    mult[i] *= dom[k]
        self.mult = mult

        TemplateStrategy.__init__(self, theta)

    def _Xmatrix(self,vect):
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
        # Note: If X is not full rank, need to modify it so that solve_triangular works
        # This doesn't impact the gradient calculations though
        # a finite difference sanity check might suggest otherwise, 
        # but a valid subgradient at theta_k = 0 is 0 due to symmetry
        #D = sparse.diags((X.diagonal()==0).astype(np.float64), format='csr')
        return X, XT

    def loss_and_grad(self):
        d = len(self.domain)
        A = np.arange(2**d)
        mult = self.mult
        dphi = self.dphi # TODO: needs to be set
        theta = self.get_params()

        delta = np.sum(theta)**2
        ddelta = 2*np.sum(theta)
        theta2 = theta**2
        Y, YT = Xmatrix(theta2)
        params = Y.dot(theta2)
        X, XT = Xmatrix(params)
        phi = spsolve_triangular(X, theta2, lower=False)
        # Note: we should be multiplying by domain size here if we want total squared error
        ans = np.dot(phi, dphi)
#        ans = np.sum([phi[b]*np.dot(mult[A|b],weights**2) for b in range(2**d)])
#        dphi = np.array([np.dot(weights**2, mult[A|b]) for b in range(2**d)])
        dXvect = -spsolve_triangular(XT, dphi, lower=True)
        # dX = outer(dXvect, phi)
        dparams = np.array([np.dot(dXvect[A&b]*phi, mult[A|b]) for b in range(2**d)])
        dtheta2 = YT.dot(dparams)
        dtheta = 2*theta*dtheta2
        return delta*ans, delta*dtheta + ddelta*ans

# (df / dtheta_k) = sum_ij (df / d_Aij) (dA_ij / theta_k)
  
def KronPIdentity(ns, ps):
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
    RangeTemplate(16, start=8, branch=2) builds a strategy template with four augmented queries that have structural zeros everywhere except in the interval indicated below:
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

if __name__ == '__main__':
    S = PIdentity(4, 16)
    W = np.random.rand(16,16)
    WtW = W.T.dot(W)
    obj, grad = S.loss_and_grad(WtW)
    print grad.shape
