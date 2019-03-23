import numpy as np
from workload import AllRange
from utility import *
from IPython import embed
import time
import matplotlib.pyplot as plt
from scipy import optimize

def augmented_optimization(WtW, B0):
    """ 
    """
    k, n = B0.shape

    logs = []

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
        logs.append(loss)
        return loss, grad.flatten()

    bnds = [(0,None)]*(k*n)

    opts = { 'ftol' : 1e-4 }
    res=optimize.minimize(loss_and_grad,x0=B0.flatten(),jac=True,method='L-BFGS-B',bounds=bnds,options=opts)
    A = np.vstack([np.eye(n), res.x.reshape(k,n)])
    return A / A.sum(axis=0), res, logs


R64 = AllRange(64)
queries = R64.queries ** 2

WtW = np.kron(R64.WtW, R64.WtW)

t0 = time.time()
err1d = []
best = np.inf
for i in range(250):
    A1, res, _ = augmented_optimization(R64.WtW, np.random.rand(4,64))
    err1d.append(res.fun)
    if res.fun < best:
        best = res.fun
        print 'new best', i, np.sqrt(best**2 / queries)
        bestA1 = A1    
t1 = time.time()
print 'done with 1d optimizations'

err1d = np.sqrt(np.minimum.accumulate(err1d)**2 / queries)
best = np.sqrt(best**2 / queries)
time1d = np.linspace(0, t1-t0, err1d.size)

A1 = np.kron(bestA1, bestA1)
print rootmse(WtW, A1, queries) # 15.6

t0 = time.time()
A2, res, err2d = augmented_optimization(WtW, np.random.rand(128, 4096))
t1 = time.time()

print rootmse(WtW, A2, queries) # 14.3
err2d = np.sqrt(np.array(err2d) / queries)
time2d = np.linspace(0, t1-t0, err2d.size)

identity = rootmse(R64.WtW, np.eye(64), R64.queries)**2
plt.plot([0, t1-t0], [identity, identity], label='Identity') 
plt.plot([0, t1-t0], [best, best], label='Kronecker')
plt.plot(time2d, err2d, label='General Purpose')
plt.ylim(0, 2*identity)
plt.xlabel('Time (s)')
plt.ylabel('RMSE')
plt.title('2D Range Queries (64 x 64)')
plt.legend()

plt.savefig('quality_vs_time.png')

plt.show()

embed()
