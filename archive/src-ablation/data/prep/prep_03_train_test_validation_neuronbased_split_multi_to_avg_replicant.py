import pandas as pd
import random
import numpy as np
import csv


dataset_path = '../../../../dataset/Alternative-Splicing/'  
fold_index = 'nocontrol-fold01'  ## be careful dont change it unless you are sure

file_name = '2025-02-01-13-40-05'
date = '12Feb2025_multireplicant-'+fold_index  
full_raw_data = pd.read_csv(dataset_path+'main/'+file_name+'_main_data.csv')
no_control = True

event_id_list = full_raw_data['event_id'].tolist()
event_id_list_set = set(event_id_list)

if no_control:
    event_id_list_set_no_c = [item for item in event_id_list_set if not item.startswith('C')]
    event_id_list_set = event_id_list_set_no_c


# sample by average replicant form all of the replicant of a neuron 
for i, event_id in enumerate(event_id_list_set):

    event_i_raw_data = full_raw_data[full_raw_data['event_id'] == event_id] 
    event_i_raw_neuron = set(event_i_raw_data['neuron'].tolist())  

    # Group by the 'neuron' column and randomly select one sample per group 
    # import pdb;pdb.set_trace()
    sampled_df_i = event_i_raw_data #.groupby('neuron').sample(n=1)

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

    
    if i == 0:
        print(sampled_raw_data_i['PSI']-mean_of_events_per_neurons[event_id])



# compute delta psi and add to the file 
delta_psi = []
for index, row in sampled_raw_data.iterrows():  
    # import pdb;pdb.set_trace()
    delta_psi.append(row['PSI']-mean_of_events_per_neurons[row['event_id']])
    
sampled_raw_data['DELTA PSI'] = delta_psi



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
 
    for index, row in sampled_raw_data.iterrows():   
        
        random_number = random.uniform(0, 1) 

        if random_number < 0.65:
            train_data.append(row)

        elif (random_number > 0.65) and (random_number < 0.80):
            valid_data.append(row) 
        else:
            test_data.append(row)
 
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



 

# def uniqe_event_tyepe():
#     event_ids = []
#     for index, row in df.iterrows():   
#         event_ids.append(row['event_id'])
    
#     event_ids_list = list(set(event_ids))

#     print(f'len(event_ids) {len(event_ids)}')
#     print(f'len(event_ids_list) {len(event_ids)}')
    
#     # First, shuffle the list to ensure random distribution
#     random.shuffle(event_ids_list)

#     # Calculate the indices for splits
#     total_len = len(event_ids_list)
#     train_end = int(total_len * 0.7)
#     test_end = train_end + int(total_len * 0.15)

#     # Split the list
#     train_genes = event_ids_list[:train_end]
#     test_genes = event_ids_list[train_end:test_end]
#     valid_genes = event_ids_list[test_end:]

#     import pdb;pdb.set_trace()

#     # Data to be written to the CSV file
#     train_data = [
#         ["event_id", "neuron", "nb_reads", "PSI", "gene_id", "p_seq_len", "sj_seq_len"], 
#     ]
#     valid_data = [
#         ["event_id", "neuron", "nb_reads", "PSI", "gene_id", "p_seq_len", "sj_seq_len"], 
#     ]
#     test_data = [
#         ["event_id", "neuron", "nb_reads", "PSI", "gene_id", "p_seq_len", "sj_seq_len"], 
#     ]
    
#     for index, row in df.iterrows():   
#         if row['event_id'] in train_genes:
#             train_data.append(row)

#         elif row['event_id'] in valid_genes:
#             valid_data.append(row)

#         elif row['event_id'] in test_genes:
#             test_data.append(row)
 
#     csv_file_path = dataset_path+'5Dec2024_EventType/2024-12-04-11-35-55_train_data.csv' 
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(train_data)

#     csv_file_path = dataset_path+'5Dec2024_EventType/2024-12-04-11-35-55_valid_data.csv' 
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(valid_data)

#     csv_file_path = dataset_path+'5Dec2024_EventType/2024-12-04-11-35-55_test_data.csv' 
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(test_data)



 