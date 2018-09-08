import workload
import templates
import numpy as np

# create a Kronecker product workload from dense matrix building blocks

domain = (10,25)

identity1 = workload.Matrix( np.eye(10) )
identity2 = workload.Matrix( np.eye(25) )
total = workload.Matrix( np.ones((1,10)) )
prefix = workload.Matrix( np.tril(np.ones((25,25))) )

W1 = workload.Kron([identity1, identity2])
W2 = workload.Kron([total, prefix])
W = workload.Concat([W1, W2])

# find a Kronecker product strategy by optimizing the workload
ps = [2,2] # parameter for P-Identity strategies
strategy = templates.KronPIdentity(domain, ps)

strategy.optimize(W)

# get the sparse representation of the optimized strategy
A = strategy.sparse_matrix()

# Skip for Lapalce Mechanism, Round for Geometric Mechanism
A = np.round(A*1000) / 1000.0

print(A)
