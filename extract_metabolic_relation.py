import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


# Parse arguments
parser = argparse.ArgumentParser(description='The abundance matrix: NCBI taxonomy id as rows and samples names as columns, final row is cancer_label')
parser.add_argument('-input', default=None,help='import abundance matrix(NCBI taxonomy id as rows and samples names as columns, final row is cancer_label')

args = parser.parse_args()
df = pd.read_csv(args.input,sep = ',',index_col = 0)
print ('The number of species is', df.shape[0]-1)
print ('The number of samples is', df.shape[1])
species_list = df.index.values[:len(df)-1].tolist()
species_list = map(lambda x: str(x), species_list)         # transfer the species id into str

# output file "*_metabolic_relation.tsv"
path = os.path.dirname(args.input)
path1 = os.path.dirname(path)
file_name=(args.input).split('/')[-1].split('.')[0]
file_1 = path+'/'+ file_name + "_njs16.csv"
file_2 = path+'/'+ file_name + "_njs16_norm.txt"
file_3 = path+'/'+ file_name + "_metabolic_matrix.csv"
file_4 = path+'/'+ file_name + "_metabolic_relation.tsv"
file_5 = path+'/'+ file_name + "_species_list.tsv"

# Need to be checked
df_njs16 = pd.read_csv(path1 + '/' + 'NJS16_metabolic_relation.txt',sep = '\t')

# obtain the contained species and metabolic compound in NJS16
df_njs16 = df_njs16[df_njs16['taxonomy ID'].isin(species_list)]

df_njs16.to_csv(file_1,sep = '\t', header = 0, index = 0 )

# normalize the data: each compound is represented as a row
f = open(file_1)
w = open(file_2,"w")
y = f.readline()
while y:
    y = y.strip()
    lis = y.split('\t')
    # print (lis)
    if '&' in lis[0] and ',' in lis[1]:
        lis_1 = lis[0].split('&')
        lis_2 = lis[1].split(',')
        for j in range(len(lis_2)):
            w.write(str(lis_1[0]) + '\t' + lis_2[j] + '\t' + lis[2] + '\n')
            w.write("Production (export)" + '\t' + lis_2[j] + '\t' + lis[2] + '\n')
    elif '&' in lis[0] and ',' not in lis[1]:
        lis_1 = lis[0].split('&')
        w.write(str(lis_1[0]) + '\t' + lis[2] + '\t' + lis[2] + '\n')
        w.write("Production (export)" + '\t' + lis[2] + '\t' + lis[2] + '\n')
        
    elif '&' not in lis[0] and ',' in lis[1]:
        lis_2 = lis[1].split(",")
        for j in range(len(lis_2)):
                w.write(str(lis[0] + '\t' + lis_2[j] + '\t' + lis[2] + '\n')) 
    else:
        w.write(str(lis[0] + '\t' + lis[1] + '\t'  + lis[2] + '\n')) 
    y = f.readline()
    
f.close()
w.close()

# all relations are considered as equivalent
p = open(file_2)
w = open(file_4,"w")

dictionary={}         # compound and species list. key is compound, and value is species
x = p.readline()
while x:
    x = x.strip()
    lis = x.split('\t')
    dictionary.setdefault(lis[1],[]).append(lis[2])
    x = p.readline()

list_species_pair = []         # move replicates of species-species relations
for i, j in dictionary.items():
    if len(dictionary[i]) != 1:
        for k in range(len(j)):
            for k_1 in range(len(j)):
                if set([j[k],j[k_1]]) not in list_species_pair:
                    list_species_pair.append(set([j[k],j[k_1]]))
                    w.write(str(j[k])+"\t"+str(j[k_1])+"\n")

p.close()
w.close()

# transfer into a metabolic relation matrix
species = df.index.values[:len(df)-1]
species_list = pd.DataFrame(species)
species_list.to_csv(file_5, index = 0, header = 0)
species = species.astype('int')
species_metabolites = pd.read_csv(file_4, sep = '\t', index_col = 0)
species1 = species_metabolites.index.values
species2 = species_metabolites.values
species2 = species2.flatten()
matrix = np.zeros((len(species), len(species)))
for i,j in enumerate(species):
    for m,n in enumerate(species1):
        if n == j:
            p = species2[m]
            q = np.argwhere(species == p)#species.index(p)
            if i != q:
                matrix[i,q] = 1
                matrix[q,i] = 1

df3 = pd.DataFrame(matrix, index=list(species), columns=list(species))
df3.to_csv(file_3, sep = ",") 
print ('The final metabolic relations among species and species is in', file_3)

os.remove(file_1)
os.remove(file_2)
os.remove(file_4)


