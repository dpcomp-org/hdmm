import numpy as np
import pickle
from workload import *
from scalability_running import krons
import utility
from IPython import embed
import optimize
import pandas as pd

results = pd.DataFrame(columns=['dataset', 'workload', 'epsilon', 'Identity', 'Privbayes', 'PMM'])

for eps in [0.1, 1.0]:

    ##################
    ## Experiment 1 ##
    ##################

    X, Xs = pickle.load(open('/home/ryan/Desktop/privbayes_data/CPS2-%.1f.pkl' % eps, 'rb'))
    diffs = []
    for est in Xs:
        diffs.append((X - est).flatten()) 

    # CPS dataset:
    # Domain = Income, Age, Marital, Race, Sex

    # Adult Dataset
    # Age, education, race, sex, hours-per-week

    W = Kron([AllRange(50), AllRange(100), Marginal(7), Marginal(4), Marginal(2)])
    #W = Kron([Identity(n) for n in [50, 100, 7, 4, 2]])
    #W = Kron([Identity(50), Identity(100), Total(7), Total(4), Total(2)])

    q = W.queries
    ps = [4,8,1,1,1]
    restarts = 50
    As = optimize.restart_kron(W, restarts, ps)
    I = [np.eye(n) for n in W.domain]

    eye = W.expected_error(I, eps=eps)
    opt = W.expected_error(As, eps=eps)
    low, high = W.average_error_ci(diffs)
    unif = W.squared_error(X.flatten() - X.sum() / np.prod(X.shape))

    print 'CPS, W1, %.1f, %.2f, %.2f, (%.2f, %.2f), %.2f' % (eps, np.sqrt(opt/q), np.sqrt(eye/q), np.sqrt(low/q), np.sqrt(high/q), np.sqrt(unif/q))

    ####################
    ### Experiment 2 ###
    ####################

    blocks = [AllRange(50), AllRange(100), Identity(7), Identity(4), Identity(2)]
    base = [Total(50), Total(100), Total(7), Total(4), Total(2)]
    Ws = []
    for i in range(5):
        for j in range(i+1,5):
            tmp = list(base)
            tmp[i] = blocks[i]
            tmp[j] = blocks[j]
            Ws.append(Kron(tmp))

    W = Concat(Ws)
    q = W.queries

    ps = [4,8,1,1,1]
    restarts = 25
    As = optimize.restart_union_kron(W, restarts, ps)
    eye = W.expected_error(I, eps=eps)
    opt = W.expected_error(As, eps=eps)
    low, high = W.average_error_ci(diffs)
    unif = W.squared_error(X.flatten() - X.sum() / np.prod(X.shape))

    print 'CPS, W2, %.1f, %.2f, %.2f, (%.2f, %.2f), %.2f' % (eps, np.sqrt(opt/q), np.sqrt(eye/q), np.sqrt(low/q), np.sqrt(high/q), np.sqrt(unif/q))

    ##################
    ## Experiment 3 ##
    ##################    

    X, Xs = pickle.load(open('/home/ryan/Desktop/privbayes_data/adult1-%.1f.pkl' % eps, 'rb'))
    diffs = []
    for est in Xs:
        diffs.append((X - est).flatten()) 
    
    W = Kron([AllRange(75), Prefix(16), Marginal(5), Marginal(2), Prefix(20)])
    q = W.queries   
 
    ps = [6, 3, 1, 1, 3]
    restarts = 50
    As = optimize.restart_kron(W, restarts, ps)
    I = [np.eye(n) for n in W.domain]

    eye = W.expected_error(I, eps=eps)
    opt = W.expected_error(As, eps=eps)
    low, high = W.average_error_ci(diffs)
    unif = W.squared_error(X.flatten() - X.sum() / np.prod(X.shape))

    print 'Adult, W1, %.1f, %.2f, %.2f, (%.2f, %.2f), %.2f' % (eps, np.sqrt(opt/q), np.sqrt(eye/q), np.sqrt(low/q), np.sqrt(high/q), np.sqrt(unif/q))

    ##################
    ## Experiment 4 ##
    ##################

    blocks = [AllRange(75), Prefix(16), Identity(5), Identity(2), Prefix(20)]
    base = [Total(n) for n in W.domain]
    Ws = []
    for i in range(5):
        for j in range(i+1,5):
            tmp = list(base)
            tmp[i] = blocks[i]
            tmp[j] = blocks[j]
            Ws.append(Kron(tmp))
    W = Concat(Ws)
    q = W.queries   
 
    ps = [6,3,1,1,3]
    restarts = 25
    As = optimize.restart_union_kron(W, restarts, ps)
    eye = W.expected_error(I, eps=eps)
    opt = W.expected_error(As, eps=eps)
    low, high = W.average_error_ci(diffs)
    unif = W.squared_error(X.flatten() - X.sum() / np.prod(X.shape))

    print 'Adult, W2, %.1f, %.2f, %.2f, (%.2f, %.2f), %.2f' % (eps, np.sqrt(opt/q), np.sqrt(eye/q), np.sqrt(low/q), np.sqrt(high/q), np.sqrt(unif/q))

