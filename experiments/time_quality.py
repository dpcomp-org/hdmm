import numpy as np
import workload
import optimize
from IPython import embed
import utility

dom = (8,8,8,8,8,8)
n = np.prod(dom)
d = len(dom)

M = workload.LowDimMarginals(dom, d//2)

p = 128
#A1, _ = optimize.augmented_optimization(M.WtW, np.random.rand(p, n))

W6D = M.subworkloads()
W3D = []
W2D = []
for A, B, C, D, E, F in W6D:
    AB = np.kron(A,B)
    CD = np.kron(C,D)
    EF = np.kron(E,F)
    ABC = np.kron(AB,C)
    DEF = np.kron(D, EF)
    W3D.append([AB,CD,EF])
    W2D.append([ABC, DEF])

As = optimize.union_kron(W6D, [np.random.rand(2, 8) for i in range(6)])
err = utility.squared_error_union_kron(W6D, As)
print err

As = optimize.union_kron(W3D, [np.random.rand(6, 8**2) for i in range(3)])
err = utility.squared_error_union_kron(W3D, As)
print err

As = optimize.union_kron(W2D, [np.random.rand(18, 8**3) for i in range(2)])
err = utility.squared_error_union_kron(W2D, As)
print err

embed()
