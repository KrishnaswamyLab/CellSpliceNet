import pandas as pd
import random
import numpy as np
import csv


dataset_path = '../../../../dataset/Alternative-Splicing/'  
# fold_index = 'nocontrol-fold02'  ## be careful dont change it unless you are sure
fold_index = ''
file_name = '2025-02-01-13-40-05'
date = '12Jan2026_single_genebased'+fold_index  
full_raw_data = pd.read_csv(dataset_path+'main/'+file_name+'_main_data.csv')
no_control = True

event_id_list = full_raw_data['event_id'].tolist()
event_id_list_set = set(event_id_list)

if no_control:
    event_id_list_set_no_c = [item for item in event_id_list_set if not item.startswith('C')]
    event_id_list_set = event_id_list_set_no_c


# sample one replicant form all of the replicant of a neuron 
for i, event_id in enumerate(event_id_list_set):
    
    
    event_i_raw_data = full_raw_data[full_raw_data['event_id'] == event_id] 
    event_i_raw_neuron = set(event_i_raw_data['neuron'].tolist()) 
    

    # Group by the 'neuron' column and randomly select one sample per group 
    sampled_df_i = event_i_raw_data.groupby('neuron').sample(n=1)

    if i == 0: 
        sampled_raw_data = sampled_df_i
    else:
        sampled_raw_data = pd.concat([sampled_raw_data, sampled_df_i], axis=0)
    
sampled_raw_data.reset_index(drop=True, inplace=True)


# needed for computing delta psi
mean_of_events_per_neurons = {}
for i, event_id in enumerate(event_id_list_set):  
    sampled_raw_data_i = sampled_raw_data[sampled_raw_data['event_id']==event_id]  
    mean_of_events_per_neurons[event_id] = np.mean(sampled_raw_data_i['PSI'].tolist())

    # import pdb;pdb.set_trace()
    # if i == 0:
    #     print(sampled_raw_data_i['PSI']-mean_of_events_per_neurons[event_id])



# compute delta psi and add to the file 
delta_psi = []
threshold = 0.0001
mask_higher_values = []
for index, row in sampled_raw_data.iterrows():  

    delta_psi_i = (1+row['PSI'])-mean_of_events_per_neurons[row['event_id']]

    if abs(delta_psi_i)<threshold:
        mask_higher_values.append(False)
    else:
        mask_higher_values.append(True)
  
    delta_psi.append(delta_psi_i)
    

sampled_raw_data['DELTA PSI'] = delta_psi

#filter out lower deltapsi values
sampled_raw_data = sampled_raw_data[mask_higher_values]




def split_genes_genelevel(gene_ids, seed=42, train_frac=0.65, val_frac=0.15, test_frac=0.20):
    genes = list(set(map(str, gene_ids)))  # ensure Python str + unique
    rng = np.random.default_rng(seed)
    rng.shuffle(genes)

    n = len(genes)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac * n))

    train_genes = set(genes[:n_train])
    val_genes   = set(genes[n_train:n_train + n_val])
    test_genes  = set(genes[n_train + n_val:])  # remainder

    return train_genes, val_genes, test_genes


def event_tyepe():
    event_ids = []
    for index, row in sampled_raw_data.iterrows():   
        event_ids.append(row['event_id'])
     
    print(f'len(event_ids) {len(event_ids)}')
    print(f'len(event_ids_list) {len(event_ids)}') 
 
    train_data = [
        ["event_id", "neuron", "neuron_replicate", "nb_reads", "PSI", "gene_id", "p_seq_len",
        "sj_seq_len", "exon_start", "exon_end", "intron_start", "intron_end", "gene_length", "DELTA PSI"], 
    ]
    valid_data = [
        ["event_id", "neuron", "neuron_replicate", "nb_reads", "PSI", "gene_id", "p_seq_len",
        "sj_seq_len", "exon_start", "exon_end", "intron_start", "intron_end", "gene_length", "DELTA PSI"], 
    ]
    test_data = [
        ["event_id", "neuron", "neuron_replicate", "nb_reads", "PSI", "gene_id", "p_seq_len",
        "sj_seq_len", "exon_start", "exon_end", "intron_start", "intron_end", "gene_length", "DELTA PSI"], 
    ]

    gene_ids = sampled_raw_data['gene_id'].tolist()
    train_genes, val_genes, test_genes = split_genes_genelevel(gene_ids, seed=42, train_frac=0.65, val_frac=0.15, test_frac=0.20)
    
 
    for index, row in sampled_raw_data.iterrows(): 

        if row['gene_id'] in train_genes:  
            train_data.append(row)
        elif row['gene_id'] in val_genes:  
            valid_data.append(row)
        elif row['gene_id'] in test_genes:  
            test_data.append(row)
        else:
            raise ValueError(f"gene_id {gid} not found in train/val/test gene sets")

    csv_file_path = dataset_path+date+'/'+file_name+'_train_data.csv' 
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_data)

    csv_file_path = dataset_path+date+'/'+file_name+'_valid_data.csv' 
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(valid_data)

    csv_file_path = dataset_path+date+'/'+file_name+'_test_data.csv' 
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_data)

event_tyepe()



  