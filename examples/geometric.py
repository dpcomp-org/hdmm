from experiments.census_workloads import CensusSF1
import templates
import utility

if __name__ == '__main__':
    sf1 = CensusSF1()
    
    ns = sf1.domain
    ps = [1,1,6,1,8]
    kron = templates.KronPIdentity(ns, ps)
    
    kron.optimize(sf1)

    As = [S.A for S in kron.strategies]

    print sf1.rootmse(As)
    
    for d in [1,2,3]:
        tmp = utility.discretize(As, digits=d)
        print d, sf1.rootmse(tmp)
