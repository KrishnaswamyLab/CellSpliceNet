import pandas as pd
import random
import csv


dataset_path = '../../../../dataset/Alternative-Splicing/'

file_name = '2025-02-01-13-40-05'
date = '01Feb2025_EventType-NoControl' 
df = pd.read_csv(dataset_path+'main/'+file_name+'_main_data.csv')
 
for index, row in df.iterrows():   
    random_number = random.uniform(0, 1)


 
def event_tyepe():
    event_ids = []
    for index, row in df.iterrows():   
        event_ids.append(row['event_id'])
     
    print(f'len(event_ids) {len(event_ids)}')
    print(f'len(event_ids_list) {len(event_ids)}') 
 
    train_data = [
        ["event_id", "neuron", "neuron_replicate", "nb_reads", "PSI", "gene_id", "p_seq_len",
        "sj_seq_len", "exon_start", "exon_end", "intron_start", "intron_end", "gene_length"], 
    ]
    valid_data = [
        ["event_id", "neuron", "neuron_replicate", "nb_reads", "PSI", "gene_id", "p_seq_len",
        "sj_seq_len", "exon_start", "exon_end", "intron_start", "intron_end", "gene_length"], 
    ]
    test_data = [
        ["event_id", "neuron", "neuron_replicate", "nb_reads", "PSI", "gene_id", "p_seq_len",
        "sj_seq_len", "exon_start", "exon_end", "intron_start", "intron_end", "gene_length"], 
    ]
      
    for index, row in df.iterrows():   
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



 