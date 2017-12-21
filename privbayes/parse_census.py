import pandas as pd

data = pd.read_csv('hh_persons0.csv')

data['RACE'] = data.RACWHT + 2*data.RACAIAN + 4*data.RACASN + 8*data.RACBLK + 16*data.RACNHPI + 32*data.RACSOR
cols = ['SEX', 'isHISP', 'RACE', 'RELP','AGEP']
data = data[cols]

data.to_csv('census.dat', sep=' ', header=False, index=False)
