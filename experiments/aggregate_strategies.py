import numpy as np
import pickle
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workload import *
import utility


base = '/home/rmckenna/work1/strategies/'

def aggregate_kron():
    workloads = ['2dunion-kron', '2drangeid-kron', '2dprefixid-kron']
    for dom in [32, 64, 128, 256, 512, 1024]:
        As = None
        best = np.inf
        for file in glob.glob('%s/%s-%d*.pkl' % (base, workload, dom)):
            ans = pickle.load(open(file, 'rb'))
            error = ans['log'][-1]
            if error < best:
                best = error
                As = ans['As']
        np.savez('%s/%s-%d.npz' % (base, workload, dom), *As)
 
def aggregate_general():
    workloads = {}
    workloads['width25'] = lambda n: WidthKRange(n, 25)
    workloads['all-range'] = AllRange
    workloads['prefix'] = Prefix
    workloads['permuted-range'] = AllRange
    workloads['2drange'] = lambda n: Kron([AllRange(n), AllRange(n)])
    workloads['2dprefix'] = lambda n:Kron([Prefix(n), Prefix(n)])
    workloads['2dunion']=lambda n:Concat([Kron([AllRange(n), Total(n)]), Kron([Total(n), AllRange(n)])])
    workloads['2drangeid']=lambda n:Concat([Kron([AllRange(n), Identity(n)]), Kron([Identity(n), AllRange(n)])])

    for dom in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        for workload, matrix in workloads.items():
            if workload in ['2drange', '2dprefix', '2dunion', '2drangeid'] and dom > 64:
                continue
            W = matrix(dom) 
            best = np.inf
            eye = utility.rootmse(W.WtW, np.eye(np.prod(W.domain)), W.queries)
            tmax = 0
            meta = []
            for file in glob.glob('%s/%s-%d*.pkl' % (base,workload, dom)):
                res = pickle.load(open(file, 'rb'))
                obj = res['res'].fun
                meta.append((obj, res['time'], res['log']))
                if obj < best:
                    best = obj
                    bestA = res['A']
                tmax = max(tmax, res['time'])
                log = np.array(res['log'])
                log = np.sqrt(log / W.queries)
                time = np.linspace(0, res['time'], len(log))
                plt.plot(time, log)
            print '%s, %d, %.2f, %.2f, %.1f' % (workload, dom, eye, np.sqrt(best / W.queries), tmax)
            plt.plot([0, tmax], [eye, eye], label='Identity')
            plt.ylim(0, 2*eye)
            plt.title('%s: %d' % (workload, dom))
            plt.savefig('plots/%s-%d.png' % (workload, dom))
            np.save('%s/%s-%d.npy' % (base, workload, dom), bestA)
            pickle.dump(meta, open('%s/%s-%d-meta.pkl' % (base, workload, dom), 'wb'))

if __name__ == '__main__':
    aggregate_general()
#    aggregate_kron()

