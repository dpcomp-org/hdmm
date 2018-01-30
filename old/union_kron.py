import autograd.numpy as np
from autograd import grad

def pinv_vjp(g, ans, vs, gvs, A):
    A1 = np.linalg.pinv(A)
    In = np.eye(A.shape[1])
    Im = np.eye(A.shape[0])
    term1 = -np.dot(A1, np.dot(g.T, A1))
    term2 = np.dot(np.dot(A1, A1.T), np.dot(g, Im - np.dot(A, A1)))
    term3 = np.dot(In - np.dot(A1, A), np.dot(g, np.dot(A1.T, A1)))
    return (term1 + term2 + term3).T

np.linalg.pinv.defvjp(pinv_vjp)

def kron_obj(As, Ws):
    # As is a l x d table
    # Ws is a k x d table
    L = len(As)
    K = len(Ws)
    D = len(As[0])
    
    #deltas = sum([reduce(np.kron, [np.sum(A, axis=0)[:,None] for A in kron]) for kron in As])
    #delta = np.max(deltas)**2
    #delta = np.sum(eps)**2
    delta = 1.0
    
    Bs = [[np.linalg.pinv(A/np.sum(A, axis=0)) for A in kron] for kron in As]
    V = [[None for _ in range(K)] for _ in range(L)]
    for l in range(L):
        for k in range(K):
            v = [None for _ in range(D)]
            for d in range(D):
                #A = As[l][d] / np.sum(As[l][d], axis=0)
                X = np.dot(Ws[k][d], Bs[l][d])
                # check to make sure strategy supports workload
                v[d] = np.sum(X**2, axis=1)[:,None]
                #if not np.allclose(np.dot(X, A), Ws[k][d]): print 'checkpt'
            V[l][k] = reduce(np.kron, v).flatten()
            
    V2 = np.array([np.concatenate(vs) for vs in V])# / eps[:,None]**2
    
    s = np.sum(1.0/V2, axis=0)
    f = np.sum(1.0 / s)
    return delta*f

def kron_obj_new(As, Ws):
    L = len(As)
    K = len(Ws)
    D = len(As[0])
    
    Ms = [[A / np.sum(A, axis=0) for A in kron] for kron in As]
    Bs = [[np.linalg.inv(np.dot(A.T, A)) for A in kron] for kron in Ms]
    V = [[None for _ in range(K)] for _ in range(L)]
    for l in range(L):
        for k in range(K):
            v = [None for _ in range(D)]
            for d in range(D):
                X = np.dot(Ws[k][d], Bs[l][d])
                Y = np.dot(X, Ws[k][d].T)
                v[d] = np.diag(Y)[:,None]
            V[l][k] = reduce(np.kron, v).flatten()
            
    V = np.array([np.concatenate(vs) for vs in V])
    
    s = np.sum(1.0/V, axis=0)
    f = np.sum(1.0 / s)
    return f * len(As)**2


if __name__ == '__main__':
    from experiments.census_workloads import CensusSF1
    from scipy import optimize
    
    sf1 = CensusSF1().project_and_merge([[0],[1],[2],[4]])

    Ws = [[S.W for S in K.workloads] for K in sf1.workloads]
    ps = [1,1,6,10]
    L = 2
    As = [[np.vstack([np.eye(n), np.random.rand(p,n)]) for p, n in zip(ps, sf1.domain)] for _ in range(L)]
    D = len(As[0])

    def vect_to_mats(params):
        idx = 0
        ans = []
        for _ in range(L):
            Ai = []
            for n, p in zip(sf1.domain, ps):
                stop = idx+n*(n+p)
                Ai.append(params[idx:stop].reshape(n+p, n))
                idx = stop
            ans.append(Ai)
        return ans

    def mats_to_vect(As):
        vects = []
        for i in range(L):
            vects.append(np.concatenate([A.flatten() for A in As[i]]))
        return np.concatenate(vects)

    gradient1 = grad(kron_obj_new, argnum=0)
    #gradient2 = grad(kron_obj, argnum=1)
    id_err = kron_obj_new([[np.eye(n) for n in sf1.domain]], Ws)

    def loss_and_grad(params):
        #eps = params[:2]
        As = vect_to_mats(params)
        #eps = params[:2]
        ans = kron_obj_new(As, Ws)
        dAs = gradient1(As, Ws)
        #deps = gradient2(As, eps, Ws)
        dparams = mats_to_vect(dAs)
        print id_err / ans #, [np.linalg.cond(A) for A in As[0]]
        #print As[0][0] / As[0][0].sum(axis=0)
        #print ans, params.sum(), np.sum([[np.sum(A) for A in Ai] for Ai in As])
        return ans, dparams

    #print kron_obj(As, Ws)
    #print grad(kron_obj)(As, Ws)

    #eps = np.ones(2)
    params = mats_to_vect(As)
    bounds = [(0, None)] * params.size
    res = optimize.minimize(loss_and_grad, x0=params, method='L-BFGS-B', jac=True, bounds=bounds)
    
