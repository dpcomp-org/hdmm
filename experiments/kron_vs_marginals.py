import numpy as np
import utility
import workload
import strategy_opt
import optimize

marginals = workload.LowDimMarginals([4]*6, 3)

best = np.inf
for i in range(100):
    res, err = strategy_opt.datacubes_optimization(marginals)
    best = min(best, err[0])

print best
print err[1:]

