import numpy as np
import templates
import workload

class ParametricMM:

    def __init__(self, W, x, eps, seed=0):
        self.domain = W.domain
        self.W = W
        self.x = x
        self.eps = eps
        self.prng = np.random.RandomState(seed)

    def optimize(self, restarts = 25):
        W = self.W
        if type(W.domain) is tuple: # kron or union kron workload
            ns = W.domain

            ps = [max(1, n//16) for n in ns]
            kron = templates.KronPIdentity(ns, ps)
            optk = kron.restart_optimize(W, restarts)

            marg = templates.Marginals(ns)
            optm = marg.restart_optimize(W, restarts)

            # multiplicative factor puts losses on same scale
            if optk['loss'] < optm['loss']*np.prod(ns):
                self.strategy = kron
            else:
                self.strategy = marg
        else:
            n = W.domain
            pid = templates.PIdentity(max(1, n//16), n)
            optp = pid.restart_optimize(W, restarts)
            self.strategy = pid
           
    def run(self):
        A = self.strategy.strategy()
        A1 = self.strategy.inverse()
        delta = self.strategy.sensitivity()
        noise = self.prng.laplace(loc=0.0, scale=delta/self.eps, size=A.shape[0])
        self.ans = A.dot(self.x) + noise
        self.xest = A1.dot(self.ans)
        return self.xest 
 
