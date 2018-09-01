from workload import *

def __race1():
    # single race only, two or more races aggregated
    # binary encoding: 1 indicates particular race is checked
    race1 = np.zeros((7, 64))
    for i in range(6):
        race1[i, 2**i] = 1.0
    race1[6,:] = 1.0 - race1[0:6].sum(axis=0)
    return Matrix(race1)

def __race2():
    # all settings of race, k races for 1..6, two or more races
    race2 = np.zeros((63+6+1, 64))
    for i in range(1,64):
        race2[i-1,i] = 1.0
        ct = bin(i).count('1') # number of races
        race2[62+ct, i] = 1.0
    race2[63+6] = race2[64:63+6].sum(axis=0) # two or more races
    return Matrix(race2) 

def __white():
    white = np.zeros((1, 64))
    white[0,1] = 1.0
    return Matrix(white)

def __isHispanic():
    return Matrix(np.array([[1,0]]))

def __notHispanic():
    return Matrix(np.array([[0,1]]))

def __adult():
    adult = np.zeros((1, 115))
    adult[0, 18:] = 1.0
    return Matrix(adult)

def __age1():
    ranges = [0, 5, 10, 15, 18, 20, 21, 22, 25, 30, 35, 40, 45, 50, 55, 60, 62, 65, 67, 70, 75, 80, 85, 115]
    age1 = np.zeros((len(ranges)-1, 115))
    for i in range(age1.shape[0]):
        age1[i, ranges[i]:ranges[i+1]] = 1.0
    return Matrix(age1)

def __age2():
    age2 = np.zeros((20, 115))
    age2[:20,:20] = np.eye(20)
    return Matrix(age2)

def __age3():
    # more range queries on age
    age3 = np.zeros((103, 115))
    age3[:100, :100] = np.eye(100)
    age3[100,100:105] = 1.0
    age3[101,105:110] = 1.0
    age3[102,110:] = 1.0
    return Matrix(age3)

def CensusKifer():
    return Kron([IdentityTotal(2), IdentityTotal(2), IdentityTotal(64), IdentityTotal(17), AllRange(115)])

def CensusSF1(geography=False):
    P1 = Kron([Total(2), Total(2), Total(64), Total(17), Total(115)])
    P3a = Kron([Total(2), Total(2), __race1(), Total(17), Total(115)])
    P3b = P1
    P4a = Kron([Total(2), Identity(2), Total(64), Total(17), Total(115)])
    P4b = P1
    P5a = Kron([Total(2), Identity(2), __race1(), Total(17), Total(115)])
    P5b = Kron([Total(2), IdentityTotal(2), Total(64), Total(17), Total(115)])
    P8a = Kron([Total(2), Total(2), __race2(), Total(17), Total(115)])
    P8b = P1
    P9a = Kron([Total(2), Identity(2), Total(64), Total(17), Total(115)])
    P9b = Kron([Total(2), __notHispanic(), __race2(), Total(17), Total(115)])
    P9c = P1
    P10a = Kron([Total(2), Total(2), __race2(), Total(17), __adult()])
    P10b = Kron([Total(2), Total(2), Total(64), Total(17), __adult()])
    P11a = Kron([Total(2), Identity(2), Total(64), Total(17), __adult()])    
    P11b = Kron([Total(2), __notHispanic(), __race2(), Total(17), __adult()])    
    P11c = P10b
    P12a = Kron([Identity(2), Total(2), Total(64), Total(17), __age1()])
    P12b = Kron([IdentityTotal(2), Total(2), Total(64), Total(17), Total(115)])
    P12_a = Kron([Identity(2), Total(2), __race1(), Total(17), __age1()])
    P12_b = Kron([IdentityTotal(2), Total(2), __race1(), Total(17), Total(115)])
    P12_c = Kron([Identity(2), __isHispanic(), Total(64), Total(17), __age1()])
    P12_d = Kron([IdentityTotal(2), __isHispanic(), Total(64), Total(17), Total(115)])
    P12_e = Kron([Identity(2), __notHispanic(), __white(), Total(17), __age1()])
    P12_f = Kron([IdentityTotal(2), __notHispanic(), __white(), Total(17), Total(115)])
    PCT12a = Kron([Identity(2), Total(2), Total(64), Total(17), __age3()])
    PCT12b = P12b
    PCT12_a = Kron([Identity(2), Total(2), __race1(), Total(17), __age3()])
    PCT12_b = Kron([IdentityTotal(2), Total(2), __race1(), Total(17), Total(115)])
    PCT12_c = Kron([Identity(2), __isHispanic(), Total(64), Total(17), __age3()])
    PCT12_d = Kron([IdentityTotal(2), __isHispanic(), Total(64), Total(17), Total(115)])
    PCT12_e = Kron([Identity(2), __notHispanic(), __race1(), Total(17), __age3()])
    PCT12_f = Kron([IdentityTotal(2), __notHispanic(), __race1(), Total(17), Total(115)])
    workloads = [P1,P3a,P3b,P4a,P4b,P5a,P5b,P8a,P8b,P9a,P9b,P9c,P10a,P10b,P11a,P11b,P11c,P12a,P12b,P12_a,P12_b,P12_c,P12_d,P12_e,P12_f,PCT12a,PCT12b,PCT12_a,PCT12_b,PCT12_c,PCT12_d,PCT12_e,PCT12_f]
    if geography:
        M = IdentityTotal(51)
        workloads = [Kron(W.workloads + [M]) for W in workloads]
    return Concat(workloads) 
    

def CensusSF1Big(geography=True, reallybig=False):
    M = IdentityTotal(51)
    I = Identity(51)
    T = Total(51)
    sf1 = CensusSF1(reallybig and geography)
    geography = geography and not reallybig
    workloads = []
    for sub in sf1.workloads:
        matrices = [S.W for S in sub.workloads]
        for combo in itertools.product(*matrices):
            subs = [Matrix(q[None,:]) for q in combo]
            if geography:
                workloads.append(Kron(subs + [I]))
                workloads.append(Kron(subs + [T]))
            else: 
                workloads.append(Kron(subs))
    return Concat(workloads) 

def CensusSF1Approx():
    R1 = Total(64) + __race1()
    R2 = Total(64) + __race2()
    A1 = Total(115) + __age1()
    A3 = Total(115) + __age3() 
 
    P1 = Kron([Total(2), Total(2), Total(64), Total(17), Total(115)])
    P3 = Kron([Total(2), Total(2), R1, Total(17), Total(115)])
    P4 = Kron([Total(2), IdentityTotal(2), Total(64), Total(17), Total(115)])
    P5 = Kron([Total(2), IdentityTotal(2), R1, Total(17), Total(115)])
    P8 = Kron([Total(2), Total(2), R2, Total(17), Total(115)])
    P9 = Kron([Total(2), IdentityTotal(2), R2, Total(17), Total(115)])
    P10 = Kron([Total(2), Total(2), R2, Total(17), __adult()])
    P11 = Kron([Total(2), IdentityTotal(2), R2, Total(17), __adult()])
    P12 = Kron([IdentityTotal(2), IdentityTotal(2), R1, Total(17), A1])
    PCT12 = Kron([IdentityTotal(2), IdentityTotal(2), R1, Total(17), A3])
    return Concat([P1, P3, P4, P5, P8, P9, P10, P11, P12, PCT12]) 

def CensusSF1Projected():
    sf1 = CensusSF1()
    sub = [None]*5
    for i in range(5):
        sub[i] = sf1.project_and_merge([[i]])
    return Kron(sub)

def CensusPL94():
    P1 = Kron([Total(2), Total(2), Total(64), Total(17), Total(115)])
    P8a = Kron([Total(2), Total(2), __race2(), Total(17), Total(115)])
    P8b = P1
    P9a = Kron([Total(2), Identity(2), Total(64), Total(17), Total(115)])
    P9b = Kron([Total(2), __notHispanic(), __race2(), Total(17), Total(115)])
    P9c = P1
    P10a = Kron([Total(2), Total(2), __race2(), Total(17), __adult()])
    P10b = Kron([Total(2), Total(2), Total(64), Total(17), __adult()])
    P11a = Kron([Total(2), Identity(2), Total(64), Total(17), __adult()])     
    P11b = Kron([Total(2), __notHispanic(), __race2(), Total(17), __adult()])
    P11c = P10b
    return Concat([P8a,P8b,P9a,P9b,P9c,P10a,P10b,P11a,P11b,P11c])

def CensusSF1_split(geography=False):
    sf1 = CensusSF1(geography)
    return Concat(sf1.workloads[:18]), Concat(sf1.workloads[18:])
 

