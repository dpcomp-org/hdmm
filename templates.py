import numpy as np
from scipy import optimize

class TemplateStrategy:
    def __init__(self, rows, cols, theta0=None):
        self.rows = rows
        self.cols = cols
        if theta0 is None:
            self.theta = np.random.rand(rows*cols)
        else:
            self.theta = theta0

    def set_params(self, theta):
        self.theta = theta

    @property
    def A(self):
        B = self.theta.reshape(self.rows, self.cols)
        return B / B.sum(axis=0)

    def sensitivity(self):
        return np.abs(self.A).sum(axis=0).max()

    def loss_and_grad(self, WtW):
        # TODO(ryan): sensitivity?
        B = self.theta.reshape(self.rows, self.cols)
        scale = B.sum(axis=0)
        A = B / scale
        AtA = A.T.dot(A)
        AtA1 = np.linalg.pinv(AtA)
        M = WtW.dot(AtA1)
        dX = -AtA1.dot(M)
        dA = 2*A.dot(dX)
        dB = (dfA*scale - (B*dfA).sum(axis=0)) / scale**2
        return np.trace(M), dB.flatten()

    def optimize(self, WtW):
        init = self.theta
        bnds = [(0,None)] * init.size
        
        def obj(theta):
            self.set_params(theta)
            return self.loss_and_grad(WtW)

        res = optimize.minimize(obj, init, jac=True, method='L-BFGS-B', bounds=bnds)
        return res

class PIdentity(TemplateStrategy):
    def __init__(self, p, n):
        theta0 = np.random.rand(p*n)
        self.p = p
        TemplateStrategy.__init__(self, p+n, n, theta0)
    
    @property
    def A(self):
        I = np.eye(self.cols)
        B = self.theta.reshape(self.p, self.cols)
        A = np.vstack([I, B])
        return A / A.sum(axis=0)

    def loss_and_grad(self, WtW):
        p, n = self.p, self.cols

        B = np.reshape(self.theta, (p,n))
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

if __name__ == '__main__':
    S = PIdentity(4, 16)
    W = np.random.rand(16,16)
    WtW = W.T.dot(W)
    obj, grad = S.loss_and_grad(WtW)
    print grad.shape

