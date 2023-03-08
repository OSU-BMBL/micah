print ('Import packages...')

import time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
import sys
from warnings import filterwarnings
filterwarnings("ignore")
from torch_geometric.data import Data
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from scipy import interp    
import torch.utils.data as data
import argparse
import os
sys.path.append(r"./pyHGT")
from utils import *
from data import *
from model import *



seed=0

random.seed(seed)
#np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

#torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Training GNN on species_sample graph')
parser.add_argument('-epoch', type=int, default=500)
# Result
parser.add_argument('-input_abundance', default= None,
                      help='The address of abundance matrix.')
parser.add_argument('-input_metabolism', default= None,
                      help='The address of metabolic matrix.')
parser.add_argument('-input_phylogeny', default= None,
                      help='The address of phylogenetic matrix.')
# Feature extration
parser.add_argument('-num', type=float, default=0.9,
                    help='the num of training data')
parser.add_argument('-reduction', type=str, default='AE',
                    help='the method for feature extraction, pca, raw') 
                     
parser.add_argument('-in_dim', type=int, default=256,
                    help='Number of hidden dimension (AE)')
                    
# GAE       
parser.add_argument('-kl_coef', type=float, default=0.00005,              
                    help='coefficient of regular term')
parser.add_argument('-gamma', type=float, default=2.5,
                    help='coefficient of focal loss')
parser.add_argument('-lr', type=float, default=0.003,
                     help='learning rate')
parser.add_argument('-n_hid', type=int, default=128,
                    help='Number of hidden dimension')
parser.add_argument('-n_heads', type=int, default=8,
                     help='Number of attention head')
parser.add_argument('-n_layers', type=int, default=2,
                    help='Number of GNN layers')
parser.add_argument('-dropout', type=float, default=0,
                    help='Dropout ratio')
parser.add_argument('-layer_type', type=str, default='hgt',
                    help='the layer type for GAE')                
parser.add_argument('-loss', type=str, default='cross',
                    help='the loss for GAE') 
                    
parser.add_argument('-cuda', type=int, default=1,
                    help='cuda 0 use GPU0 else cpu ') 
           
parser.add_argument('-rep', type=str, default='iT',
                    help='precision truncation')                     

parser.add_argument('-AEtype', type=int, default=1,
                    help='AEtype1: embedding node autoencoder 2:HGT node autoencode')   
                    


args = parser.parse_args()


file0='Micah1_'+'_lr_'+str(args.lr)+str(args.epoch)+'_kl_para_'+str(args.kl_coef)+'_gamma_'+str(args.gamma)

path = os.path.dirname(args.input_dir1)
file_name=(args.input_dir1).split('/')[-1].split('.')[0]
att_file1 = path+'/'+ file_name + "_attention.csv"
path1 = path+'/temp'
model_dir1=path+'/temp/'+'hgt_parameter/'
model_dir2=path+'/temp/'+'AE_parameter/'
model_dir3=path+'/temp/'+'AE_loss/'
model_dir4=path+'/temp/'+'hgt_loss/'
model_dir5=path+'/temp/'+'roc_point/'
model_dir6=path+'/temp/'+'test_index/'
if os.path.exists(path1) == False:
    os.mkdir(path+'/temp')
if os.path.exists(model_dir1) == False:
    os.mkdir(path+'/temp'+'/hgt_parameter')
if os.path.exists(model_dir2) == False:
    os.mkdir(path+'/temp'+'/AE_parameter')
if os.path.exists(model_dir3) == False:
    os.mkdir(path+'/temp'+'/AE_loss')
if os.path.exists(model_dir4) == False:
    os.mkdir(path+'/temp'+'/hgt_loss')
if os.path.exists(model_dir5) == False:
    os.mkdir(path+'/temp'+'/roc_point')
if os.path.exists(model_dir6) == False:
    os.mkdir(path+'/temp'+'/test_index')
    
    

def  load_data(path, sep, col_name, row_name):
    if col_name == True and row_name == True:
        gene_cell_matrix = pd.read_csv(path, sep=sep, index_col=0)
    gene_cell_matrix1 = gene_cell_matrix.dropna(axis=0)
    gene_cell_matrix = gene_cell_matrix1[0:len(gene_cell_matrix1)-1]
    cutoff = int(gene_cell_matrix.shape[1]*0.005)
    cut_matrix = gene_cell_matrix.astype(float).gt(0).sum(axis=1)
    cut_species = cut_matrix[cut_matrix>cutoff].index.tolist()
    gene_cell_matrix = gene_cell_matrix.loc[cut_species,]
    if gene_cell_matrix.shape[0] == gene_cell_matrix1.shape[0]-1:
        cell_label = gene_cell_matrix1[len(gene_cell_matrix1)-1:].T
        #cell_label = pd.get_dummies(cell_label.T)
        gene_cell = gene_cell_matrix.values
        gene_cell = gene_cell[0:len(gene_cell_matrix1)-1,:]
        gene_name = gene_cell_matrix.index.values
        cell_name = gene_cell_matrix.columns.values
    if gene_cell_matrix.shape[0] < gene_cell_matrix1.shape[0]-1:
        gene_name = gene_cell_matrix.index.values
        cell_name = gene_cell_matrix.columns.values
        cell_label = gene_cell_matrix1[len(gene_cell_matrix1)-1:].T
        gene_cell_matrix = np.array(gene_cell_matrix).astype('float')
        sum_count = np.sum(gene_cell_matrix,axis=0)
        matrix1 = []
        for i,j in zip(gene_cell_matrix, sum_count):
            matrix1.append(i/j)
        gene_cell_matrix = matrix1
        gene_cell_matrix = np.array(gene_cell_matrix)
        gene_cell_matrix = pd.DataFrame(gene_cell_matrix)
        gene_cell = gene_cell_matrix.values
        gene_cell = gene_cell[0:len(gene_cell_matrix1)-1,:]

    print("The number of species is {}, and the number of samples is {}.".format(gene_cell.shape[0],gene_cell.shape[1]))
    return(gene_cell_matrix1, gene_cell_matrix, cell_label,gene_cell,  gene_name, cell_name)


def split_cell_train_test(data, name, train_ratio):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(np.arange(len(data)))
    train_set_size = int(len(data) * train_ratio)
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:len(data)]
    train_data = []
    train_name = []
    test_data = []
    test_name = []
    for i in train_indices:
        train_data.append(data[i])
        train_name.append(name[i])
    for i in test_indices :
        test_data.append(data[i])
        test_name.append(name[i])
    return train_data, test_data, train_name, test_name
def shuffle(x1,y1,z1):
    x = []
    y = []
    z = []
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    for i in index:
        x.append(x1[i])
        y.append(y1[i])
        z.append(z1[i])
    return x,y,z

class AE(nn.Module):
    def __init__(self, dim):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.tanh(self.fc1(x))            
        return F.tanh(self.fc2(h1))
	

    def decode(self, z):
        h3 = F.tanh(self.fc3(z))
        return F.tanh(self.fc4(h3))


    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):                       
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   
            print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 

        self.gamma = gamma

    def forward(self, preds_softmax, preds_logsoft,labels):
        self.alpha = self.alpha.to(preds_softmax.device)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


start_time = time.time()
print('---0:00:00---scRNA starts loading.')
gene_cell_matrix1, gene_cell_matrix,  cell_label,gene_cell, gene_name, cell_name = load_data(args.input_dir1,sep=",",col_name = True, row_name = True)
Label_transform = LabelEncoder()
cell_label_num = Label_transform.fit_transform(cell_label)
num_type = len(Label_transform.classes_)
gene_cell_matrix = gene_cell_matrix.astype('float')
gene_cell = gene_cell.astype('float')
#num_type = len(np.unique(cell_label.values))
cell_set = {int(k):[] for k in cell_label_num}
cell_name_set = {int(k):[] for k in cell_label_num}
for j, i in enumerate(cell_label_num):
    for k in cell_set:
        if int(i) ==k:
            cell_set[k].append(gene_cell[:,j]) 
            cell_name_set[k].append(cell_label.index.values[j])
weight = []
for i in range(num_type):
    label_count = len(cell_set[i])
    weight.append(1-(label_count/gene_cell_matrix.shape[1]))

all_cell = np.array(gene_cell)
all_cell = all_cell.T
all_label = np.array(cell_label_num)
all_name = np.array(cell_name)

cuda = args.cuda #'cpu'#-1   
if cuda == 0:
    device = torch.device("cuda:" + "0")   
    print("cuda>>>")
else:
    device = torch.device("cpu")
print(device)



k = 10
kf = StratifiedKFold(n_splits=k,shuffle=False)  
oversampling =  RandomOverSampler(sampling_strategy='not majority',random_state=0)
fold_train_index = [] 
fold_test_index = [] 
fold_train_cell = {i:[] for i in range(k)}
fold_train_label = {i:[] for i in range(k)}
fold_train_name = {i:[] for i in range(k)}
fold_name = {i:[] for i in range(k)}
fold_test_cell = {i:[] for i in range(k)}
fold_test_label = {i:[] for i in range(k)}
fold_test_name = {i:[] for i in range(k)}
fold_over_cell = {i:[] for i in range(k)}
fold_over_label = {i:[] for i in range(k)}
fold_over_name = {i:[] for i in range(k)}
fold_train_encoded2 = dict()
fold_test_encoded2 = dict()
fold_gene = dict()
for  train_index , test_index in kf.split(all_cell,all_label):
    fold_train_index.append(train_index)
    fold_test_index.append(test_index)
fold_train_index = np.array(fold_train_index).reshape([k,-1])
fold_test_index = np.array(fold_test_index).reshape([k,-1])
for i in range(k):
    n = list(fold_train_index[i])
    m = list(fold_test_index[i])
    fold_train_cell[i], fold_train_label[i],fold_train_name[i] = all_cell[n], all_label[n], all_name[n]
    fold_test_cell[i], fold_test_label[i],fold_test_name[i] = all_cell[m], all_label[m], all_name[m]
    fold_train_cell[i] = np.squeeze(fold_train_cell[i])
    fold_train_label[i]= np.squeeze(fold_train_label[i])
    fold_train_name[i] = np.squeeze(fold_train_name[i])
    fold_test_cell[i]= np.squeeze(fold_test_cell[i])
    fold_test_label[i]= np.squeeze(fold_test_label[i])
    fold_test_name[i] = np.squeeze(fold_test_name[i])
    fold_over_cell[i], fold_over_label[i] = oversampling.fit_resample(fold_train_cell[i], fold_train_label[i])
    fold_over_name[i] = list(1 for i in range(len(fold_over_label[i])-len(fold_train_label[i])))
    fold_name[i] = np.concatenate((fold_train_name[i],fold_over_name[i]),axis=0)
    fold_train_cell[i], fold_train_label[i],fold_train_name[i] = shuffle(fold_over_cell[i], fold_over_label[i], fold_name[i])
    fold_train_cell[i]=np.array(fold_train_cell[i])
    fold_train_label[i]=np.array(fold_train_label[i])
    fold_train_name[i]=np.array(fold_train_name[i])

train_cell = fold_train_cell[4]
train_label = fold_train_label[4]
train_cell_name = fold_train_name[4]
test_cell = fold_test_cell[4]
test_label = fold_test_label[4]
test_cell_name = fold_test_name[4]

print ('Autoencoder is trainning...')

train_cell = train_cell.T
test_cell = test_cell.T
l1 = []
l2 = []
l3 = []
if (args.reduction == 'AE'):
    gene = torch.tensor(train_cell,dtype=torch.float32).to(device)
    if train_cell.shape[0]<5000:
        ba = train_cell.shape[0]
    else:
        ba = 5000
    loader1=data.DataLoader(gene,ba)  

    EPOCH_AE = 250
    model = AE(dim=train_cell.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    for epoch in range(EPOCH_AE):
        embedding1=[]	
        for _,batch_x in enumerate(loader1)	:

            decoded, encoded = model(batch_x)
        #encoded1 , decoded1 = Coder2(cell)
            loss = loss_func(batch_x,decoded)
            l1.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            embedding1.append(encoded)	
        # print('Epoch :', epoch,'|','train_loss:%.12f'%loss.data)
    if gene.shape[0]%ba!=0:     
        torch.stack(embedding1[0:int(gene.shape[0]/ba)])           
        a=torch.stack(embedding1[0:int(gene.shape[0]/ba)])
        a=a.view(ba*int(gene.shape[0]/ba),args.in_dim)           
        encoded=torch.cat((a,encoded),0)                

    else:
        encode = torch.stack(embedding1)
        encoded=encode.view(gene.shape[0],args.in_dim)

    if train_cell.shape[1]<5000:          
        ba1 = train_cell.shape[1]
    else:
        ba1 = 5000  
    cell = torch.tensor(train_cell.T,dtype=torch.float32).to(device) 
    if test_cell.shape[1]<5000:         
        ba2 = test_cell.shape[1]
    else:
        ba2 = 5000
    cell1 = torch.tensor(test_cell.T,dtype=torch.float32).to(device) 
    loader2=data.DataLoader(cell,ba1)
    loader3=data.DataLoader(cell1,ba2)
    model2 = AE(dim=train_cell.shape[0]).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    EPOCH_AE2 = 250
    for epoch in range(EPOCH_AE2):
        embedding1=[]
        embedding2=[]
        for _,batch_x in enumerate(loader2):	
            decoded2, encoded2 = model2(batch_x)
            loss = loss_func(batch_x,decoded2)
            l2.append(loss.item())
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            embedding1.append(encoded2)
            print('Epoch :', epoch,'|','train_loss:%.12f'%loss.data)
        for _,x in enumerate(loader3):
            decoded3, encoded3 = model2(x)
            test_loss = loss_func(x,decoded3)
            l3.append(test_loss.item())
            embedding2.append(encoded3)
        
    if cell.shape[0]%ba1!=0:
        torch.stack(embedding1[0:int(cell.shape[0]/ba1)])
        a=torch.stack(embedding1[0:int(cell.shape[0]/ba1)])
        a=a.view(ba1*int(cell.shape[0]/ba1),args.in_dim)
        encoded2=torch.cat((a,encoded2),0)
        #encoded2.shape
    else:
        encode = torch.stack(embedding1)
        encoded2=encode.view(cell.shape[0],args.in_dim)
    encode2 = torch.stack(embedding2)
    encoded3 = encode2.view(cell1.shape[0],args.in_dim) 
    
plt.figure()
plt.plot(l1,'r-')
plt.title('species loss per iteration')
plt.savefig(model_dir3+ 'species_'+str(file0)+'.png')
plt.figure()
plt.plot(l2,'r-')
plt.plot(l3,'g-')
plt.title('train-test loss per iteration')
plt.savefig(model_dir3+'sample_'+str(file0)+'.png')
if (args.reduction == 'raw'):          
      encoded = torch.tensor(gene_cell,dtype=torch.float32).to(device)
      encoded2 =torch.tensor(np.transpose(gene_cell),dtype=torch.float32).to(device)


if args.input_dir2 != None:
    gene_name11 =[str(i) for i in gene_name.astype('int64')]
    gene_name12 =[int(i) for i in gene_name]
    species_species = pd.read_csv(args.input_dir2,sep=",",index_col=0)
    species_species = species_species.loc[:,gene_name11]
    species_species = species_species.loc[gene_name12,:]
    species_name = species_species.columns.values
    species_matrix = species_species.values
    #species_matrix = np.zeros((1218,1218))
    g12 = np.nonzero(species_matrix)[0]
    c22 = np.nonzero(species_matrix)[1]   
    edge12 = list(g12)
    edge22 = list(c22)
else:
    g12 = np.zeros([gene_cell_matrix.shape[0],])
    c22 = np.zeros([gene_cell_matrix.shape[0],])   
    edge12 = list(g12)
    edge22 = list(c22)
    
    
if args.input_dir3 != None:
    species_species1 = pd.read_csv(args.input_dir3,sep=",",index_col=0)
    species_species1 = species_species1.loc[:,gene_name11]
    species_species1 = species_species1.loc[gene_name12,:]
    species_name1 = species_species1.columns.values
    species_matrix1 = species_species1.values
    #species_matrix = np.zeros((1218,1218))
    g13 = np.nonzero(species_matrix1)[0]
    c23 = np.nonzero(species_matrix1)[1] 
    edge13 = list(g13)
    edge23 = list(c23)
else:
    g13 = np.zeros([gene_cell_matrix.shape[0],])
    c23 = np.zeros([gene_cell_matrix.shape[0],])   
    edge13 = list(g13)
    edge23 = list(c23)

#target_nodes = np.arange(train_cell.shape[0]+train_cell.shape[1])           
# gene cell 
g11 = np.nonzero(train_cell)[0]                                                
c21 = np.nonzero(train_cell)[1]+train_cell.shape[0]                              
edge11 = list(g11)
edge21 = list(c21)
#edge1 = edge11+edge12
#edge2 = edge21+edge22
#edge_index = torch.tensor([edge1, edge2], dtype=torch.long)
x={'gene': torch.tensor(encoded, dtype=torch.float),                      
	    'cell': torch.tensor(encoded2, dtype=torch.float),}
edge_index_dict={('gene','g_c','cell'): torch.tensor([g11, c21], dtype=torch.long),
                 ('cell','c_g','gene'): torch.tensor([c21, g11], dtype=torch.long),
                 ('gene','g_g','gene'):torch.tensor([g12, c22], dtype=torch.long),
                 ('gene1','g_g','gene1'):torch.tensor([g13, c23], dtype=torch.long)
                }

edge_reltype={('gene','g_c','cell'):  torch.tensor([g11, c21]).shape[1],
                ('cell','c_g','gene'):  torch.tensor([c21, g11]).shape[1],
                ('gene','g_g','gene'):  torch.tensor([g12, c22]).shape[1],
                ('gene1','g_g','gene1'):  torch.tensor([g13, c23]).shape[1]
                }
num_nodes_dict={
    'gene': train_cell.shape[0],
    'cell': train_cell.shape[1]}
data=Data(edge_index_dict=edge_index_dict,edge_reltype=edge_reltype,num_nodes_dict=num_nodes_dict, x=x)


a = np.nonzero(train_cell)[0]
b = np.nonzero(train_cell)[1]
node_type = list(np.zeros(train_cell.shape[0]))+list(np.ones(train_cell.shape[1]))
#node_type1 = pd.DataFrame(node_type)
#node_type1.to_csv('/fs/ess/PCON0022/yuhan/HGT/result/check_repeat/'+'node_type'+str(file0)+'.csv', sep=",")
node_type = torch.LongTensor(node_type)
#node_type = node_type.to(device)
node_feature = []
for t in ['gene','cell']:
    if args.reduction != 'raw':
        node_feature += list(x[t])
    else:
        node_feature[t_i] = torch.tensor(x[t],dtype=torch.float32).to(device)
if (args.reduction != 'raw'):
    node_feature = torch.stack(node_feature)
    node_feature = torch.tensor(node_feature,dtype=torch.float32)
    node_feature=node_feature.to(device)
#node_feature1 = node_feature.detach().numpy()
#process_encoded1 = pd.DataFrame(node_feature1)
#process_encoded1.to_csv('/fs/ess/PCON0022/yuhan/HGT/result/check_repeat/'+'process_encoded'+str(file0)+'.csv', sep=",")
    #print(node_feature)
edge_index1 = data['edge_index_dict'][('gene',  'g_c',  'cell')]
edge_index2 = data['edge_index_dict'][('cell','c_g','gene')]
edge_index3 = data['edge_index_dict'][('gene',  'g_g',  'gene')]
edge_index4 = data['edge_index_dict'][('gene1',  'g_g',  'gene1')]
edge_index = torch.cat((edge_index1,edge_index2,edge_index3,edge_index4),1)
#edge_index = torch.cat((edge_index1,edge_index2),1)
edge_type = list(np.zeros(len(edge_index1[1])))+list(np.ones(len(edge_index2[1])))+list(2 for i in range(len(edge_index3[1])))+list(3 for i in range(len(edge_index4[1])))
edge_time = torch.LongTensor(list(np.zeros(len(edge_index[1]))))
edge_type = torch.LongTensor(edge_type)
edge_index = torch.LongTensor(edge_index.numpy())



test_g11=np.nonzero(test_cell)[0]
test_c21=np.nonzero(test_cell)[1]+test_cell.shape[0]
test_edge11 = list(test_g11)
test_edge21 = list(test_c21)
#test_edge1 = test_edge11+edge12
#test_edge2 = test_edge21+edge22
#test_edge_index = torch.tensor([test_edge1, test_edge2], dtype=torch.long)
test_x={'gene': torch.tensor(encoded, dtype=torch.float),
   'cell': torch.tensor(encoded3, dtype=torch.float),} #batch of gene all cells
#edge_index_dict2 = {('gene','g_c','cell'): torch.tensor([g11, c21], dtype=torch.long)}
edge_index_dict2 = {('gene','g_c','cell'): torch.tensor([test_g11, test_c21], dtype=torch.long),
                    ('cell','c_g','gene'): torch.tensor([test_c21, test_g11], dtype=torch.long),
                    ('gene','g_g','gene'): torch.tensor([g12, c22], dtype=torch.long),
                    ('gene1','g_g','gene1'):torch.tensor([g13, c23], dtype=torch.long)
                    }
                   
edge_reltype2 = {('gene','g_c','cell'):  torch.tensor([test_g11, test_c21]).shape[1],
                 ('cell','c_g','gene'): torch.tensor([test_c21, test_g11]).shape[1],
                 ('gene','g_g','gene'):  torch.tensor([g12, c22]).shape[1],
                 ('gene1','g_g','gene1'):  torch.tensor([g13, c23]).shape[1]
                 }
                 
num_nodes_dict2 = {'gene': test_cell.shape[0], 'cell': test_cell.shape[1]}
data2 = Data(edge_index_dict=edge_index_dict2,edge_reltype=edge_reltype2,num_nodes_dict=num_nodes_dict2, x=test_x)
#a = np.nonzero(adj)[0]
#b = np.nonzero(adj)[1]
node_type8 = list(np.zeros(test_cell.shape[0]))+list(np.ones(test_cell.shape[1]))
node_type8 = torch.LongTensor(node_type8)
#node_type8 = node_type8.to(device)
node_feature2 = []
for t in ['gene','cell']:
    if args.reduction != 'raw':
        node_feature2 += list(test_x[t])
    else:
        node_feature2[t_i] = torch.tensor(test_x[t],dtype=torch.float32).to(device)
if (args.reduction != 'raw'):
    node_feature2 = torch.stack(node_feature2)
    node_feature2 = torch.tensor(node_feature2,dtype=torch.float32)
    node_feature2 = node_feature2.to(device)
test_edge_index1 = data2['edge_index_dict'][('gene',  'g_c',  'cell')]
test_edge_index2 = data2['edge_index_dict'][('cell','c_g','gene')]
test_edge_index3 = data2['edge_index_dict'][('gene',  'g_g',  'gene')]
test_edge_index4 = data2['edge_index_dict'][('gene1',  'g_g',  'gene1')]
test_edge_index = torch.cat((test_edge_index1,test_edge_index2,test_edge_index3,test_edge_index4),1)
#test_edge_index = torch.cat((test_edge_index1,test_edge_index2),1)
edge_type2 = list(np.zeros(len(test_edge_index1[1])))+list(np.ones(len(test_edge_index2[1])))+list(2 for i in range(len(test_edge_index3[1])))+list(3 for i in range(len(test_edge_index4[1])))
edge_time2 = torch.LongTensor(list(np.zeros(len(test_edge_index[1]))))
edge_type2 = torch.LongTensor(edge_type2)
test_edge_index = torch.LongTensor(test_edge_index.numpy())

np.random.seed(seed)
torch.manual_seed(seed)
#debuginfoStr('Cell Graph constructed and pruned')
#print(jobs[0])
#print(graph.years[np.random.choice(np.arange(gene_cell.shape[0]), args.batch_size, replace = False)])
if (args.reduction != 'raw'):
   gnn = GNN(conv_name = args.layer_type, in_dim = encoded.shape[1], \
          n_hid =args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = 2, num_relations = 4, use_RTE = False).to(device)    
else:
    gnn = GNN_from_raw(conv_name = args.layer_type, in_dim = [encoded.shape[1],encoded2.shape[1]], \
          n_hid =args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = 2, num_relations = 4, use_RTE = False,\
          AEtype=args.AEtype).to(device)      
classifier = Classifier(args.n_hid,num_type).to(device)

args_optimizer =  'adamw'
if args_optimizer == 'adamw':
    optimizer = torch.optim.AdamW([
                {'params': gnn.parameters()},
                {'params': classifier.parameters()}
            ],lr=args.lr)
elif args_optimizer == 'adam':
    optimizer = torch.optim.Adam(gnn.parameters(),lr = args.lr)
elif args_optimizer == 'sgd':
    optimizer = torch.optim.SGD(gnn.parameters(),lr = args.lr)
elif args_optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(gnn.parameters(),lr = args.lr)
#gnn.double()

#model, optimizer = amp.initialize(gnn, optimizer, opt_level="O1")
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=5, verbose=True)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
##index = np.argwhere(train_label<5)
##train_label1 = np.delete(train_label,index)
train_label1 = torch.LongTensor(train_label.flatten()).to(device)  ##一会加个1
test_label1 = torch.LongTensor(test_label.flatten()).to(device)
#print(type(gnn.parameters()))


loss_function = focal_loss(alpha=weight,gamma=args.gamma,num_classes=num_type).to(device)
#k=[]
train_loss_all = []
train_F1 = []
test_loss_all = []
test_F1 = []
for epoch in np.arange(args.epoch):
    gnn.train()
    classifier.train()
    L = 0
    if (args.reduction=='raw'):
        node_rep,node_decoded_embedding = gnn.forward(node_feature, node_type.to(device), \
                               edge_time.to(device), \
                               edge_index.to(device), \
                               edge_type.to(device))
    else:
            node_rep = gnn.forward(node_feature, node_type.to(device), \
                               edge_time.to(device), \
                               edge_index.to(device), \
                               edge_type.to(device))
    train_att1 = gnn.att1
    train_att2 = gnn.att2            
    if args.rep =='T':
        node_rep = torch.trunc(node_rep*10000000000)/10000000000
        if args.reduction=='raw':
            for t in types:
                t_i = node_dict[t][1]
                 #print("t_i="+str(t_i))
                node_decoded_embedding[t_i] = torch.trunc(node_decoded_embedding[t_i]*10000000000)/10000000000
        
    print(node_rep.shape)
        #print(abc)

    
    gene_matrix = node_rep[node_type==0,]
    cell_matrix = node_rep[node_type==1,]
    decoder = torch.mm(gene_matrix, cell_matrix.t())
    adj= torch.tensor(train_cell,dtype=torch.float32).to(device)
    #adj1 = np.matmul(train_cell,train_cell.T)
    #adj1 = torch.tensor(adj1,dtype=torch.float32).to(device)    
    KL_loss = F.kl_div(decoder.softmax(dim=-1).log(), adj.softmax(dim=-1), reduction='sum')
    pre_label, pre_score  = classifier.forward(cell_matrix)
    cross_loss = loss_function(pre_score,pre_label,train_label1)
    loss = args.kl_coef*KL_loss+cross_loss
    train_loss_all.append(loss.item())
    true_score = label_binarize(train_label,  classes=[i for i in range(num_type)])
    pre_score2 = pre_score.detach().numpy()
    train_pre_score = [np.argmax(i) for i in pre_score2]
    train_f1 = f1_score(train_label, train_pre_score, average='macro')
    train_F1.append(train_f1)
        
        
    L = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()          
    scheduler.step(L)
    print('Epoch :', epoch+1,'|','train_loss:%.12f'%(L))
    gnn.eval()
    classifier.eval()
    test_node_rep = gnn.forward(node_feature2, node_type8.to(device), \
                           edge_time2.to(device), \
                           test_edge_index.to(device), \
                           edge_type2.to(device))
    test_att1 = gnn.att1
    test_att2 = gnn.att2        
    if args.rep =='T':
        test_node_rep = torch.trunc(node_rep*10000000000)/10000000000
        if args.reduction=='raw':
            for t in types:
                t_i = node_dict[t][1]
                 #print("t_i="+str(t_i))
                node_decoded_embedding[t_i] = torch.trunc(node_decoded_embedding[t_i]*10000000000)/10000000000
        
    test_gene_matrix = test_node_rep[node_type8==0,]
    test_cell_matrix = test_node_rep[node_type8==1,]
    test_decoder = torch.mm(test_gene_matrix, test_cell_matrix.t())
    test_adj= torch.tensor(test_cell,dtype=torch.float32).to(device)   
    test_KL_loss = F.kl_div(test_decoder.softmax(dim=-1).log(), test_adj.softmax(dim=-1), reduction='sum')
    test_pre_label, test_pre_score  = classifier.forward(test_cell_matrix)
    test_cross_loss = loss_function(test_pre_score,test_pre_label,test_label1)
    test_loss = args.kl_coef*test_KL_loss+test_cross_loss
    test_loss_all.append(test_loss.item())
    pre_score1 = test_pre_score.detach().numpy()
    pre_score11 = [np.argmax(i) for i in pre_score1]
    test_f1 = f1_score(test_label, pre_score11, average='macro')
    test_F1.append(test_f1)
    print('Epoch :', epoch+1,'|','test_loss:%.12f'%(test_loss.item()))

        
attention1 = []
attention1_no_softmax = []
attention1.append(train_att1[:len(np.array(edge_index1[0])),:])
attention1_no_softmax.append(train_att2[:len(np.array(edge_index1[0])),:])
attention1 = attention1[0].detach().numpy()
attention1_no_softmax = attention1_no_softmax[0].detach().numpy()
gene_name1 = list(gene_name)
edge_index1 = torch.LongTensor(edge_index1).numpy()
gene_name1 = [gene_name1[i] for i in list(np.array(edge_index1[0]))]
cell_name1 = [train_cell_name[i] for i in list(np.array(edge_index1[1]-train_cell.shape[0]))]
label_name1 = [train_label[i] for i in list(np.array(edge_index1[1]-train_cell.shape[0]))]
label_name1 = Label_transform.inverse_transform(label_name1)
attention2 = []
attention2_no_softmax = []
gene_name2 = list(gene_name)
gene_name2 = [gene_name2[i] for i in list(np.array(test_edge_index1[0]))]
test_cell_name = list(test_cell_name)
cell_name2 = [test_cell_name[i] for i in list(np.array(test_edge_index1[1]-test_cell.shape[0]))]
test_label = list(test_label)
label_name2 = [test_label[i] for i in list(np.array(test_edge_index1[1]-test_cell.shape[0]))]
label_name2 = Label_transform.inverse_transform(label_name2)
attention2.append(test_att1[:len(np.array(test_edge_index1[0])),:])
attention2_no_softmax.append(test_att2[:len(np.array(test_edge_index1[0])),:])
attention2 = attention2[0].detach().numpy()
attention2_no_softmax = attention2_no_softmax[0].detach().numpy()



plt.figure()
plt.plot(train_loss_all,'r-')
plt.plot(test_loss_all,'g-')
plt.title('train-test loss per iteration')
plt.savefig(model_dir4+ 'loss_'+str(file0)+'.png')
plt.figure()
plt.plot(train_F1,'r-')
plt.plot(test_F1,'g-')
plt.title('train-test F1 per iteration')
plt.savefig(model_dir4+ 'F1_'+str(file0)+'.png')
n_classes = true_score.shape[1]   
fpr=dict()
tpr=dict()
roc_auc = dict()
#pre_score = pre_score.detach().numpy()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(true_score[:, i], pre_score2[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='b', linestyle=':', linewidth=4)
colors = ['m', 'c','r','g','y']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig(model_dir5+str(file0)+'.png')


 
if os.path.exists(model_dir1) == False:
    os.mkdir(model_dir1)
state1 = {'model_1':gnn.state_dict(),'model_2':classifier.state_dict(), 'optimizer':scheduler.state_dict(), 'epoch':epoch}
torch.save(state1,model_dir1+file0)


#pre_score1 = [np.argmax(i) for i in pre_score1]
target_names = Label_transform.inverse_transform([i for i in range(num_type)])
others = classification_report(y_true=test_label,y_pred=pre_score11,labels=[i for i in range(num_type)],target_names=target_names,output_dict=True)
others = pd.DataFrame(others).transpose()
others.to_csv(model_dir6+str(file0)+'.csv',index = 0)

gene_name = gene_name1+gene_name2
cell_name = cell_name1+cell_name2
gene_name = np.array(gene_name)
cell_name = np.array(cell_name)
over_index = np.argwhere(cell_name != "1").flatten()
cell_name = cell_name[over_index]
gene_name = gene_name[over_index]
label_name1 = list(label_name1)
label_name2 = list(label_name2)
label_name = label_name1+label_name2
label_name = np.array(label_name)
label_name = label_name[over_index]
attention = np.concatenate([attention1, attention2], axis=0)
attention = attention[over_index]
attention_no_softmax = np.concatenate([attention1_no_softmax, attention2_no_softmax], axis=0)
attention_no_softmax = attention_no_softmax[over_index]
#g = np.nonzero(adj)[0]
#c = np.nonzero(adj)[1]+adj.shape[0]
name1 = pd.DataFrame(
    gene_name, columns=['taxa_id'])
name2 = pd.DataFrame(
    cell_name, columns=['Sample'])
name3 = pd.DataFrame(
    label_name,columns=['cancer_type'])
df = pd.DataFrame(attention,columns=['attention_head_1','attention_head_2','attention_head_3','attention_head_4','attention_head_5','attention_head_6','attention_head_7','attention_head_8'])
df2 = pd.DataFrame(attention_no_softmax,columns=['attention_head_1','attention_head_2','attention_head_3','attention_head_4','attention_head_5','attention_head_6','attention_head_7','attention_head_8'])
df = pd.concat([name1, name2, name3, df], axis=1)
df2 = pd.concat([name1, name2, name3, df2], axis=1)
df.to_csv(att_file1, sep=",", index=True)  
print ('The final attention score (after softmax) is in ', att_file1)