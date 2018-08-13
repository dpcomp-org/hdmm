import numpy as np


def optimize(WtW):
    V = WtW

    accuracy = 1e-5
    max_iter_ls = 50
    max_iter_cg = 5
    theta = 1e-3
    
    beta = 0.5
    sigma = 1e-4
    n = V.shape[0]
    I = np.eye(n)
    X = I
    max_iter = 10
    V = V + theta*np.mean(np.diag(V))*I
    
    iX = I
    G = -V
    fcurr = np.sum((V*iX)**2)
    history = []

    for iter in range(1, max_iter+1):
        if iter == 1:
            D = -G
            np.fill_diagonal(D,0)
            j = 0
        else:
            D = np.zeros((n,n))
            Hx = lambda S: -iX.dot(S).dot(G) - G.dot(S).dot(iX)
            np.fill_diagonal(D, 0)
            R = -G - Hx(D)
            np.fill_diagonal(R, 0)
            P = R;
            rsold = np.sum(R**2)
            for j in range(1, max_iter_cg+1):
                Hp = Hx(P)
                alpha = rsold / np.sum(P * Hp)
                D += alpha*P
                np.fill_diagonal(D, 0)
                R -= alpha*Hp
                np.fill_diagonal(R, 0)
                rsnew = np.sum(R**2)
                if np.sqrt(rsnew) < 1e-8:
                    break
                P = R + rsnew / rsold * P
                rsold = rsnew

        delta = np.sum(D * G)
        X_old = X
        flast = fcurr
        history.append(fcurr)
        
        for i in range(1, max_iter_ls+1):
            alpha = beta**(i-1)
            X = X_old + alpha*D
            iX = np.linalg.inv(X)
            try:
                A = np.linalg.cholesky(X)
            except:
                continue
            G = -iX.dot(V).dot(iX)
            fcurr = np.sum(V * iX)
            if fcurr <= flast + alpha*sigma*delta:
                break

        print(fcurr)

        if i==max_iter_ls:
            X = X_old
            fcurr = flast
            break
        if np.abs((flast - fcurr) / flast) < accuracy:
            break

    return np.linalg.cholesky(X)

 
if __name__ == '__main__':
    n = 256
    r = np.arange(n)+1
    X = np.outer(r, r[::-1])
    WtW = np.minimum(X, X.T)
    optimize(WtW)
