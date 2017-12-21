import numpy as np
import matplotlib.pyplot as plt
from utility import squared_error

W = np.vstack([np.ones(64), np.eye(64)])

def A(c):
    return np.vstack([c*np.ones(64), (1-c)*np.eye(64)])

def f(c):
    return squared_error(W.T.dot(W), A(c)) / W.shape[0]

xs = np.linspace(0,0.4)
ys = [f(c) for c in xs]

plt.plot(xs, ys)
plt.xlabel('c')
plt.ylabel('Per-Query Expected Squared Error')
plt.show()
