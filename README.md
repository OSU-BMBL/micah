# MICAH



## Description ##

MICAH provides an explainable deep learning framework to identify cancer-associated intratumoral microbial communities. The microbial species-sample abundance matrix is required input in MICAH, while metabolic and phylogenetic relation matrices are optional inputs. The output is microbial communities composed of species with significantly high attention scores for each cancer type. 

If you have any questions or feedback, please contact Qin Ma qin.ma@osumc.edu.

## Usage ##

### 1. Dependencies 

The following packages are required for MICAH.

 - python 3.7+
 	- pytorch 1.3.0+cpu
 	- torch-geometric 1.3.2
 	- torch-scatter == 1.3.2
 	- torch-sparse == 0.4.3
 	- torch-cluster  == 1.4.5
 	- scikit-learn == 1.0.2     
 	- imbalanced-learn == 0.9.0
 - R > 4.0
 	- taxizedb (R package)

You can construct conda environment named micah and install these dependencies. 

	conda create -n micah
	conda activate micah
	conda install pytorch==1.3.0 torchvision==0.4.1 torchaudio==0.3.1 cpuonly -c pytorch
	pip install torch-scatter==1.3.2 torch-sparse==0.4.3 torch-cluster==1.4.5 torch-spline-conv==1.1.1 torch-geometric==1.3.2 -f https://data.pyg.org/whl/torch-1.3.0%2Bcpu.html
	conda install scikit-learn
	pip install imbalanced-learn
	conda install tqdm
	pip install seaborn
	pip install dill


### 2. Installation
The source code of MICAH is freely available at https://github.com/OSU-BMBL/micah. To install MICAH, you can download the zip file manually from GitHub, or use the code below in Unix.
   	 
	cd /your working path/ 
	wget https://github.com/OSU-BMBL/micah/archive/refs/heads/master.zip


Then, unzip the file and go to the folder:

	unzip master.zip && rm -rf master.zip
	cd ./micah-master


### 3. Data preparation

#### 3.1 Microbial species-sample abundance matrix

In the microbial species-sample abundance matrix, the first row represents the sample names; the first column represents microbial species indicated by NCBI taxonomy ID; and the entry represents the relative abundance of a species in the corresponding sample. The last row represents sample labels. 

We provide a microbial relative abundance matrix, **./data/tcma.csv**, as an example. You can obtain the results in our paper based on this data. 

Note: The following commands will run using this data as an example. You can change this into your own abundance matrix.

 
#### 3.2 Metabolic and phylogenetic relation matrices

The metabolic/phylogenetic relation matrix indicates whether there is a metabolic/phylogenetic relationship between two microbial species, with a value of one indicating a relationship and zero indicating no relationship. The two matrices are optional inputs for MICAH. If you do not provide them, MICAH will assess the metabolic and phylogenetic relationships among species and generate the two matrices automatically. 

Note: The name of microbial species must keep the same format as that in the abundance matrix.

<table>
	<tbody>
       <tr>
           <td>NCBI_taxid</td>
           <td>72</td>
           <td>158</td>
		   <td>199</td>
        </tr>
        <tr>
            <td align="center">72</td>
            <td align="center">0</td>
            <td align="center">1</td>
			<td align="center">0</td>
        </tr>
       <tr>
           <td align="center">158</th>
           <td align="center">1</th>
           <td align="center">0</th>
		   <td align="center">0</th>
        </tr>
       <tr>
           <td align="center">199</th>
           <td align="center">0</th>
           <td align="center">0</th>
		   <td align="center">0</th>
        </tr>
	<tbody>
 </table>


### 4. Running

#### 4.1 Assess the metabolic and phylogenetic relations
This step is to assess metabolic and phylogenetic relations among microbial species. If you have metabolic and phylogenetic relation matrices, please skip this. 


	python ./extract_metabolic_relation.py -input ./data/tcma.csv
	Rscript ./taxize.R ./data/*_species_list.tsv
	python ./phylo_matrix.py -input1 ./data/*_species_list_phy_relation.csv -input2 ./data/tcma.csv

	rm -rf ./data/*_species_list.tsv
	rm -rf ./data/*_species_list_phy_relation.csv


After these commands, you can obtain two result files with suffix metabolic\_matrix.csv and phy\_matrix.csv in the folder ./data/, representing the metabolic and phylogenetic matrices, respectively. 


#### 4.2 Train a graph attention transformer for sample classification

This step is to train a graph attention transformer for sample classification. Three files (microbial species-sample abundance matrix, the metabolic matrix, and the phylogenetic matrix) are used as input here. 


	python ./micah_HGT.py -input_abundance ./data/tcma.csv -input_metabolism ./data/*_metabolic_matrix.csv -input_phylogeny ./data/*_phy_matrix.csv 

You can obtain the attention matrix in the folder ./data/, with suffix attention.csv. This is used for identifying cancer-associated microbial communities following. 

You can also obtain the classification performance (including accuracy, precision, recall, F1_score) and loss curve of the model in the folder ./micah/data/temp.

  
#### 4.3 Output the cancer-associated microbial communities

This step is to output cancer-associated intratumoral microbial communities. The attention matrix from step 3.2 and the abundance matrix are used as input here. 

 
	python ./selection.py -input_attention ./data/*_attention.csv -input_abundance ./data/tcma.csv 

You can obtain a file with suffix **_final_taxa.tsv** in the folder ./data/. It includes identified communities from MICAH, in which the first column indicates cancer type, followed by the communities consisting of microbial species. The other file with suffix _taxa_num.csv in the folder ./data/ includes the number of samples of a certain cancer each taxa significantly contribute to. More samples means the species is more likely to be associated with the cancer type.
   
### 5. Others

#### Parameter description


-epoch: the number of iteration while training the graph attention transformer;

-lr: the learning rate while training the graph attention transformer;

-kl_coef: the coefficient of regular term in the graph attention transformer;

-gamma: the parameter of focal loss function;

-t: the threshold of species contribution significance;

-pv: the p-value threshold in species selection. 

