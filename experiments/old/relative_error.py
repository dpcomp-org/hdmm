from census_workloads import *
import optimize
import implicit
import pickle
from IPython import embed
import matplotlib.pyplot as plt

X = np.load('/home/ryan/Desktop/privbayes_data/new/numpy/census.npy')

result = {}

for geography in [False, True]:
    for eps in [0.1, 1.0]:
        
        if geography:
            dims = [[0],[1],[2],[4],[5]]
            axes = 3
            p = [1,1,8,10,1]
        else:
            dims = [[0],[1],[2],[4]]
            axes = (3,5)
            p = [1,1,8,10]

        x = X.sum(axis=axes).flatten()
        N = x.sum()
        sf1 = CensusSF1(geography=True).project_and_merge(dims)
        W = implicit.stack(*[implicit.krons(*[S.W for S in K.workloads]) for K in sf1.workloads])

        A_sf1 = optimize.restart_union_kron(sf1, 50, p)
        A1 = implicit.krons(*[np.linalg.pinv(A) for A in A_sf1])
        noise = A1.dot(np.random.laplace(loc=0, scale=1.0/eps, size=A1.shape[1]))
        err = W.dot(noise)
        ans = W.dot(x)
        result[(geography, eps)] = (ans, err)
        rel = np.abs(err) / np.maximum(W.dot(x), 100) #np.maximum(W.dot(x), 0.001*N)
        print geography, '%.1f, %.4f, %.4f' % (eps, np.median(rel), np.percentile(rel, 95))
#        tmp = np.arange(ans.max())
#        plt.plot(ans, ans+err, '.')
#        plt.plot(tmp,tmp)
#        plt.xscale('symlog')
#        plt.yscale('symlog')
#        plt.show()

pickle.dump(result, open('results/relative_error.pkl', 'wb'))
