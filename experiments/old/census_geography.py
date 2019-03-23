import glob
import zipfile
import pandas as pd
from IPython import embed
import numpy as np
from scipy.sparse import block_diag, csr_matrix

# downlaod files here: https://www.census.gov/geo/maps-data/data/baf.html
# note: connecticut and puerto rico didn't download for some reason

# each BLOCKID is a 15 digit string containing the fips code for state (2), county (3), tract (6), block (4)

zips = sorted(glob.glob('/home/ryan/Downloads/BlockAssign_ST*.zip'))

results = []

for z in zips:
    print(z)
    archive = zipfile.ZipFile(z, 'r')
    name = z[:-4].split('/')[-1] + '_CD.txt'
    df = pd.read_csv(archive.open(name), dtype=str)

    cols = ['State', 'County', 'Tract', 'Block']
    df['State'] = df.BLOCKID.str[:2]
    df['County'] = df.BLOCKID.str[2:5]
    df['Tract'] = df.BLOCKID.str[5:11]
    df['Block'] = df.BLOCKID.str[11:15]

    results.append(df[cols])

results = pd.concat(results).reset_index(drop=True)

results.to_csv('census-geography.csv')

states = results.groupby('State').size().count()
counties = results.groupby(['State', 'County']).size().groupby(level=0).size()
tracts = results.groupby(['State', 'County', 'Tract']).size().groupby(level=(0,1)).size()
blocks = results.groupby(['State', 'County', 'Tract']).size()

H1 = csr_matrix(np.ones((1, states)))
H2 = block_diag([np.ones(n) for n in counties], format='csr')
H3 = block_diag([np.ones(n) for n in tracts], format='csr')
#H4 = block_diag([np.ones(n) for n in blocks], format='csr')

# National x States matrix
# States x Counties matrix
# Counties x Tract matrix
# Tract x Block matrix


embed()
