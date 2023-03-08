import pandas as pd
import numpy as np
import math
import argparse
import os
import warnings

warnings.filterwarnings('ignore')
# Parse arguments
parser = argparse.ArgumentParser(description='the attention csv matrix')
parser.add_argument('-input_attention', default=None,help='import the attention matrix address')
parser.add_argument('-input_abundance', default=None,help='import the abundance matrix')
parser.add_argument('-t', type=int, default=3,help='input the threshold') 
parser.add_argument('-pv', type=float, default=0.05,help='input the p-value threshold')

args = parser.parse_args()
# read file(.tsv)
f = open(args.input1)
df = pd.read_csv(args.input2,sep = ',',index_col = 0)            # obtain the total number of species and obtain the output path

path = os.path.dirname(args.input2)
file_name=(args.input2).split('/')[-1].split('.')[0]
file1 = path+'/'+ file_name + "_taxa_num.csv"
file2 = path+'/'+ file_name +'_final_taxa.tsv'

# input HGT attention file
t = args.t
pv = args.pv
# print (t)
p = pd.read_csv(f,sep=',')

# obtain the dictionary consisting of diseases and corresponding samples 
disease_list =list(set(p['cancer_type'].tolist()))
value = []
# print (disease_list[1])
for i in range(len(disease_list)):
    sample_lis = []
    for j in range(p.shape[0]):
        if p.iloc[j,3] == disease_list[i]:
            sample_lis.append(p.iloc[j,2])
    sample_list = list(set(sample_lis))
    value.append(sample_list)
dictionary = dict(zip(disease_list,value))

# for each cancer type: key ;  obtain a dict: dict{cancer_type: [{taxa of each sample}];}
total_lis_taxa = []       
for key,value in dictionary.items():
    # print (key)
    lis_taxa = []
    for k in range(len(value)):
        # for each sample value[k],
        tem_p = p.loc[p['Sample']== value[k]]
        taxa_ = set()
        for j in range(4,12):
            # print (tem_p.iloc[:,j])
            a = np.array(tem_p.iloc[:,j])
            # print (a)
            lower_q=np.quantile(a,0.25,interpolation='lower')
            higher_q=np.quantile(a,0.75,interpolation='higher')
            q1_q3 =list(tem_p.iloc[:,j][(tem_p.iloc[:,j]>lower_q) & (tem_p.iloc[:,j]<higher_q)])
            mean_ = np.mean(q1_q3)
            std_ = np.std(q1_q3)
            # caculate threshold
            thre_ = mean_ + std_ * t
            taxa_j = set(tem_p['taxa_id'][tem_p.iloc[:,j]>thre_])
            taxa_ = taxa_.union(taxa_j)
        lis_taxa.append(taxa_) 
    total_lis_taxa.append(lis_taxa)   
dictionary_taxa = dict(zip(disease_list,total_lis_taxa))  

# calculate the number of each taxa: dict_taxa_num_all; the selected taxa number of each sample: dict_sample_taxa_num_all 
dict_taxa_num_t = []
sample_taxa_num_t = []
for key,value in dictionary_taxa.items():
    # print (key)
    # print (len(value))
    set_union = set()
    sample_taxa_num = []
    for i in range(len(value)):
        set_union = set_union.union(value[i])
        sample_taxa_num.append(len(value[i]))
    sample_taxa_num_t.append(sample_taxa_num)
    list_union = list(set_union)
    
    # caculate the number of each taxa for each phenotype
    list_num = []
    for k in range(len(list_union)):
        s = 0
        for j in range(len(value)):
            s = s + int(list_union[k] in value[j])
            # print (int(list_union[k] in value[j]))
        list_num.append(s)
    dict_taxa_num = dict(zip(list_union,list_num))  
    dict_taxa_num_t.append(dict_taxa_num)
    
dict_taxa_num_all = dict(zip(disease_list,dict_taxa_num_t))
dict_sample_taxa_num_all = dict(zip(disease_list,sample_taxa_num_t))
# print (dict_taxa_num_all) 
# print (dict_sample_taxa_num_all) 

df1 = pd.DataFrame(dict_taxa_num_all).fillna(0)
df1.to_csv(file1)

# caculate the threshold for each phenotype: dict_thre = {phenotype:taxa_number_threshold}
list_thre = []
for key,value in dict_sample_taxa_num_all.items(): 
    a_max = max(value)
    b_min = min(filter(lambda x: x > 0, value))
    # print (len(value))
    n = df.shape[0]- 1
    m = a_max
    a = math.factorial(n)//(math.factorial(m)*math.factorial(n-m))
    n_1 = df.shape[0]-2
    m_1 = a_max - 1 
    b = math.factorial(n_1)//(math.factorial(m_1)*math.factorial(n_1-m_1))
    rate = b/a
    # print (rate)
    m = m_1 = b_min
    a = math.factorial(n)//(math.factorial(m)*math.factorial(n-m))
    b = math.factorial(n_1)//(math.factorial(m_1)*math.factorial(n_1-m_1))
    rate_1 = b/a
    # print (rate_1)
    
    p_value = 0
    # print (len(value))
    for i in range(len(value),-1,-1):
        p_value = p_value + math.factorial(len(value))//(math.factorial(i)*math.factorial(len(value)-i)) * math.pow(rate,i) * math.pow(rate_1,len(value)-i)
        if p_value > pv:
            break
    list_thre.append(i)
dict_thre = dict(zip(disease_list,list_thre)) 
print ("The threshold of supported samples:", dict_thre.items())
# print (dict_thre)


# select the significant taxa for each phenotype
list_final = []
for key,value in dict_taxa_num_all.items():
    list_final_1 = []
    for key_1,value_1 in value.items():
        if value_1 > dict_thre[key]:
            list_final_1.append(key_1)
    list_final.append(list_final_1)
    print ('The number of selected taxa of ',key, ': ', len(list_final_1))
dict_final_taxa = dict(zip(disease_list,list_final))

# print final selected taxa into a file; as well as the number of supported samples
f = open(file2, 'w')
for header,elem in dict_final_taxa.items():
    f.write(str(header) + '\t')
    for j in range(len(elem)):
        f.write(str(elem[j]))
        f.write('\t')
    f.write('\n')
f.close()  
print ('The final selected taxa of each phenotype are in ', file2) 
print ('The number of samples with the selected taxa are in ',file1)

            

            


