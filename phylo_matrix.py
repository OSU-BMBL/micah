import numpy as np
import pandas as pd
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='transfer the phylogenic relations from Rscript to matrix')
parser.add_argument('-input1', default=None,help='import the phylogenic relations from Rscript')
parser.add_argument('-input2', default=None,help='import abundance matrix (NCBI taxonomy id as rows and samples names as columns, final row is cancer_label')

args = parser.parse_args()

# output file "*_metabolic_relation.tsv"
path = os.path.dirname(args.input2)
file_name=(args.input2).split('/')[-1].split('.')[0]
file_3 = path+'/'+ file_name + "_phy_matrix.csv"

species_species = pd.read_csv(args.input2,sep=",",index_col=0)
species = species_species.index.values[:len(species_species)-1]
species = species.astype('int')
species_phylo = pd.read_csv(args.input1,sep="\t",index_col=0)
genus = species_phylo.index.values
genus_unrepeat = np.unique(genus)
species3 = species_phylo.values
matrix1 = np.zeros((len(species),len(species)))
for i in genus_unrepeat:
    if i != 0:
        p = np.argwhere(genus == i).flatten()
        species4 = species3[p].flatten()
        q = []
        for n,m in enumerate(species):
            for k,h in enumerate(species4):
                if h == m:
                    q.append(n)
        for j in q:
            for g in q:
                if j != g:
                    matrix1[j,g] = 1    
df = pd.DataFrame(matrix1, index=list(species), columns=list(species))
df.to_csv(file_3, sep = ",")
print ('The final phylogenic relations among species and species is in', file_3)