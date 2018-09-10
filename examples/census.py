import workload
import templates
import numpy as np

# create a Kronecker product workload from dense matrix building blocks

# this is a 2d example:
domain = (10,25)

# densely represented sub-workloads in each of the dimensions
identity1 = workload.Matrix( np.eye(10) )
identity2 = workload.Matrix( np.eye(25) )
total = workload.Matrix( np.ones((1,10)) )
prefix = workload.Matrix( np.tril(np.ones((25,25))) )

# form the kron products in each dimension
W1 = workload.Kron([identity1, identity2])
W2 = workload.Kron([total, prefix])

# form the union of krons
W = workload.Concat([W1, W2])

# find a Kronecker product strategy by optimizing the workload
ps = [2,2] # parameter for P-Identity strategies
strategy = templates.KronPIdentity(domain, ps)

# run optimization
strategy.optimize(W)

# get the sparse, explicit representation of the optimized strategy
A = strategy.sparse_matrix()

# Round for Geometric Mechanism (skip this is using Laplace Mechanism)
A = np.round(A*1000) / 1000.0

print(A)
