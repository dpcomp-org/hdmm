from hdmm.workload import EkteloMatrix, Identity, Total, VStack, Kronecker, IdentityTotal
import itertools
import numpy as np
# Predicates for defining SF1 tables (Universe Persons)

def __race1():
    # single race only, two or more races aggregated
    # binary encoding: 1 indicates particular race is checked
    race1 = np.zeros((7, 64))
    for i in range(6):
        race1[i, 2**i] = 1.0
    race1[6,:] = 1.0 - race1[0:6].sum(axis=0)
    return EkteloMatrix(race1)

def __race2():
    # all settings of race, k races for 1..6, two or more races
    race2 = np.zeros((63+6+1, 64))
    for i in range(1,64):
        race2[i-1,i] = 1.0
        ct = bin(i).count('1') # number of races
        race2[62+ct, i] = 1.0
    race2[63+6] = race2[64:63+6].sum(axis=0) # two or more races
    return EkteloMatrix(race2) 

def __white():
    white = np.zeros((1, 64))
    white[0,1] = 1.0
    return EkteloMatrix(white)

def __isHispanic():
    return EkteloMatrix(np.array([[1,0]]))

def __notHispanic():
    return EkteloMatrix(np.array([[0,1]]))

def __adult():
    adult = np.zeros((1, 115))
    adult[0, 18:] = 1.0
    return EkteloMatrix(adult)

def __age1():
    ranges = [0, 5, 10, 15, 18, 20, 21, 22, 25, 30, 35, 40, 45, 50, 55, 60, 62, 65, 67, 70, 75, 80, 85, 115]
    age1 = np.zeros((len(ranges)-1, 115))
    for i in range(age1.shape[0]):
        age1[i, ranges[i]:ranges[i+1]] = 1.0
    return EkteloMatrix(age1)

def __age2():
    age2 = np.zeros((20, 115))
    age2[:20,:20] = np.eye(20)
    return EkteloMatrix(age2)

def __age3():
    # more range queries on age
    age3 = np.zeros((103, 115))
    age3[:100, :100] = np.eye(100)
    age3[100,100:105] = 1.0
    age3[101,105:110] = 1.0
    age3[102,110:] = 1.0
    return EkteloMatrix(age3)

# SF1 Table components (universe Persons)

P1 = Kronecker([Total(2), Total(2), Total(64), Total(17), Total(115)])
P3a = Kronecker([Total(2), Total(2), __race1(), Total(17), Total(115)])
P3b = P1
P4a = Kronecker([Total(2), Identity(2), Total(64), Total(17), Total(115)])
P4b = P1
P5a = Kronecker([Total(2), Identity(2), __race1(), Total(17), Total(115)])
P5b = Kronecker([Total(2), IdentityTotal(2), Total(64), Total(17), Total(115)])
P8a = Kronecker([Total(2), Total(2), __race2(), Total(17), Total(115)])
P8b = P1
P9a = Kronecker([Total(2), Identity(2), Total(64), Total(17), Total(115)])
P9b = Kronecker([Total(2), __notHispanic(), __race2(), Total(17), Total(115)])
P9c = P1
P10a = Kronecker([Total(2), Total(2), __race2(), Total(17), __adult()])
P10b = Kronecker([Total(2), Total(2), Total(64), Total(17), __adult()])
P11a = Kronecker([Total(2), Identity(2), Total(64), Total(17), __adult()])
P11b = Kronecker([Total(2), __notHispanic(), __race2(), Total(17), __adult()])
P11c = P10b
P12a = Kronecker([Identity(2), Total(2), Total(64), Total(17), __age1()])
P12b = Kronecker([IdentityTotal(2), Total(2), Total(64), Total(17), Total(115)])
P12_a = Kronecker([Identity(2), Total(2), __race1(), Total(17), __age1()])
P12_b = Kronecker([IdentityTotal(2), Total(2), __race1(), Total(17), Total(115)])
P12_c = Kronecker([Identity(2), __isHispanic(), Total(64), Total(17), __age1()])
P12_d = Kronecker([IdentityTotal(2), __isHispanic(), Total(64), Total(17), Total(115)])
P12_e = Kronecker([Identity(2), __notHispanic(), __white(), Total(17), __age1()])
P12_f = Kronecker([IdentityTotal(2), __notHispanic(), __white(), Total(17), Total(115)])
PCT12a = Kronecker([Identity(2), Total(2), Total(64), Total(17), __age3()])
PCT12b = P12b
PCT12_a = Kronecker([Identity(2), Total(2), __race1(), Total(17), __age3()])
PCT12_b = Kronecker([IdentityTotal(2), Total(2), __race1(), Total(17), Total(115)])
PCT12_c = Kronecker([Identity(2), __isHispanic(), Total(64), Total(17), __age3()])
PCT12_d = Kronecker([IdentityTotal(2), __isHispanic(), Total(64), Total(17), Total(115)])
PCT12_e = Kronecker([Identity(2), __notHispanic(), __race1(), Total(17), __age3()])
PCT12_f = Kronecker([IdentityTotal(2), __notHispanic(), __race1(), Total(17), Total(115)])

# dictionary for SF1 tables
sf1 = {}

sf1['P1'] = [P1]
sf1['P3'] = [P3a, P3b]
sf1['P4'] = [P4a, P4b]
sf1['P5'] = [P5a, P5b]
sf1['P8'] = [P8a, P8b]
sf1['P9'] = [P9a, P9b, P9c]
sf1['P10'] = [P10a, P10b]
sf1['P11'] = [P11a, P11b, P11c]
sf1['P12'] = [P12a, P12b]
sf1['P12A_I'] = [P12_a, P12_b, P12_c, P12_d, P12_e, P12_f]
sf1['PCT12'] = [PCT12a, PCT12b]
sf1['PCT12A_O'] = [PCT12_a, PCT12_b, PCT12_c, PCT12_d, PCT12_e, PCT12_f]


def build_workload(workload_keys):
    workloads = []
    for key in workload_keys:
        workloads.extend(sf1[key])
    return VStack(workloads)


def SF1_Persons():
    workload_keys = ['P1', 'P3', 'P4', 'P5', 'P8', 'P9', 'P10', 'P11', 'P12', 'P12A_I', 'PCT12', 'PCT12A_O']
    return build_workload(workload_keys)


