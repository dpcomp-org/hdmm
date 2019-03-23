import numpy as np
import utility
import workload
from dpcomp_core.algorithm import HB2D, privelet2D, QuadTree, identity
from IPython import embed
#import optimize
#import templates
import pickle
import implicit

base = '/home/ryan/Desktop/strategies'

pkl = '/home/ryan/Desktop/2dsimulations.pkl'

try: 
    estimates = pickle.load(open(pkl, 'rb'))
    trials = estimates.values()[0].shape[0]
    for key in estimates.keys():
        estimates[key] = [X.flatten() for X in estimates[key]]
    print 'estimates loaded'
except:
    estimates = {}
    trials = 50
    for dom in [32, 64, 128, 256, 512, 1024]:
        estimates[('hb', dom)] = np.zeros((trials, dom, dom))
        estimates[('wav', dom)] = np.zeros((trials, dom, dom))
        estimates[('quad', dom)] = np.zeros((trials, dom, dom))
#        estimates[('id', dom)] = np.zeros((trials, dom, dom))
        X = np.zeros((dom, dom))
        seeds = np.random.randint(0, high=1000000, size=trials)
        for i in range(trials):
            estimates[('hb', dom)][i]=HB2D.HB2D_engine().Run(None, X, np.sqrt(2), seeds[i])
            estimates[('wav',dom)][i]=privelet2D.privelet2D_engine().Run(None,X,np.sqrt(2),seeds[i])
            estimates[('quad',dom)][i]=QuadTree.QuadTree_engine().Run(None, X, np.sqrt(2), seeds[i])
#            estimates[('id',dom)][i]=identity.identity_engine().Run(None, X, np.sqrt(2), seeds[i])
            
    pickle.dump(estimates, open(pkl, 'wb'))
    print 'simulations done'

def apply_strategy(strategy, eps=np.sqrt(2)):
    diffs = [None]*trials
    # sensitivity calculation, assumes non-negative entries
    delta = strategy.rmatvec(np.ones(strategy.shape[0])).max()
    for i in range(trials):
        noise = np.random.laplace(0, 1.0/eps, size=strategy.shape[0])
        diffs[i] = implicit.sparse_inverse(strategy).dot(noise)
    return diffs

def strategy_helper(A, B):
    A1 = implicit.krons(A/2.0,B)
    A2 = implicit.krons(B,A/2.0)
    return implicit.stack(A1, A2)

workloads = {}
#workloads['2drange'] = workload.Range2D
#workloads['2dprefix'] = workload.Prefix2D
workloads['2dunion'] = workload.RangeTotal2D
workloads['2drangeid'] = workload.RangeIdentity2D
workloads['2dprefixid'] = workload.PrefixIdentity2D

print 'PMM-Kron Param, PMM-General Purpose, Identity, HB2D*, Privelet*, QuadTree*'
for name, matrix in workloads.items():
    for dom in [32, 64, 128, 256, 512, 1024]:
        A_R = np.load('%s/all-range-%d.npy' % (base, dom))
        A_P = np.load('%s/prefix-%d.npy' % (base, dom))
        I = np.eye(dom)
        T = np.ones((1,dom))
        W = matrix(dom)
        pmmb_pm = 0
        if name == '2drange':
            pmma = W.expected_error([A_R, A_R])
            pmmb = pmma
        elif name == '2dprefix':
            pmma = W.expected_error([A_P, A_P])
            pmmb = pmma
        elif name == '2dunion':
#            R = workload.AllRange(dom)
#            pmmb = 8.0 * R.expected_error(A_R)
            A = strategy_helper(A_R, T)
            diffs = apply_strategy(A)
            pmmb_ci = W.average_error_ci(diffs)
            pmmb = np.mean(pmmb_ci)

            strategy = np.load('%s/2dunion-kron-%d.npz' % (base, dom))
            A1 = strategy['arr_0']
            A2 = strategy['arr_1']
            pmma = W.expected_error([A1, A2])
        elif name == '2drangeid':
            strategy = np.load('%s/2drangeid-kron-%d.npz' % (base, dom))
            A1 = strategy['arr_0']
            A2 = strategy['arr_1']
            pmma = W.expected_error([A1, A2])
           
            A = strategy_helper(A_R, I) 
            diffs = apply_strategy(A)
            pmmb_ci = W.average_error_ci(diffs)
            pmmb = np.mean(pmmb_ci)
  
#            W1 = workload.Kron([workload.AllRange(dom), workload.Identity(dom)])
#            A1 = np.load('%s/all-range-%d.npy' % (base, dom))
#            pmmb = 8 * W1.expected_error([A1, np.eye(dom)])

        elif name == '2dprefixid':
            strategy = np.load('%s/2dprefixid-kron-%d.npz' % (base, dom))
            A1 = strategy['arr_0']
            A2 = strategy['arr_1']
            pmma = W.expected_error([A1, A2])

#            W1 = workload.Kron([workload.Prefix(dom), workload.Identity(dom)])
#            A1 = np.load('%s/prefix-%d.npy' % (base, dom))
#            pmmb = 8 * W1.expected_error([A1, np.eye(dom)])
            A = strategy_helper(A_P, I) 
            diffs = apply_strategy(A)
            pmmb_ci = W.average_error_ci(diffs)
            pmmb = np.mean(pmmb_ci)
 

        if dom <= 64 and name != '2dprefixid': 
            PMM2D = np.load('%s/%s-%d.npy' % (base, name, dom))
            pmmc = workload.Workload.expected_error(W, PMM2D)
#            A = A.dot(np.eye(dom**2))
#            pmmb_exact = workload.Workload.expected_error(W, A)
        else:
            pmmc = np.nan
        pmm = min(pmma, pmmb, pmmc)
        eye = W.expected_error([np.eye(dom), np.eye(dom)])

        def normalize(ci):
            low = np.sqrt(ci[0] / pmm)
            high = np.sqrt(ci[1] / pmm)
            return (low+high)/2.0, (high-low)/2.0

        hb = W.average_error_ci(estimates[('hb', dom)])
        wav = W.average_error_ci(estimates[('wav', dom)])
        quad = W.average_error_ci(estimates[('quad', dom)])

        row = '%s, %d,' % (name, dom)
        row += '%.2f,' % np.sqrt(pmma/pmm)
        row += '%.2f +/- %.2f,' % normalize(pmmb_ci)
#        row += ' (%.2f) ' % np.sqrt(pmmb_exact / pmm)
        row += '%.2f,' % np.sqrt(pmmc/pmm)
        row += '%.2f,' % np.sqrt(eye/pmm)
        row += '%.2f +/- %.2f,' % normalize(hb)
        row += '%.2f +/- %.2f,' % normalize(wav)
        row += '%.2f +/- %.2f' % normalize(quad)
        print row
