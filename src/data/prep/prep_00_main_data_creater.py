import torch 
import numpy as np


import sys 
path_to_modules = '../../' 
sys.path.append(path_to_modules)

torch.manual_seed(1337)
np.random.seed(42)
import pandas as pd
# from args import argparser_fn   
# from args import mkdir_fun

# from utils.plt_dataloader import plt_dataloader 
# from utils.scatter import scatter_plot

# def neuron_type_fn():
#     NEURON_TYPE_ENCODING = {
#         "AVH": 0,
#         "CAN": 1,
#         "VD": 2,
#         "AFD": 3,
#         "RIM": 4,
#         "AWC": 5,
#         "AIY": 6,
#         "RIC": 7,
#         "LUA": 8,
#         "AVE": 9,
#         "RIS": 10,
#         "PVC": 11,
#         "ADL": 12,
#         "ASK": 13,
#         "DD": 14,
#         "PVM": 15,
#         "ASG": 16,
#         "AVA": 17,
#         "NSM": 18,
#         "AVK": 19,
#         "ASER": 20,
#         "OLQ": 21,
#         "SMD": 22,
#         "DA": 23,
#         "AVM": 24,
#         "DVC": 25,
#         "VC": 26,
#         "AIN": 27,
#         "OLL": 28,
#         "ASI": 29,
#         "IL1": 30,
#         "IL2": 31,
#         "AIM": 32,
#         "SMB": 33,
#         "RMD": 34,
#         "BAG": 35,
#         "AWB": 36,
#         "AVL": 37,
#         "AWA": 38,
#         "AVG": 39,
#         "ASEL": 40,
#         "VB": 41,
#         "PHA": 42,
#         "PVD": 43,
#         "RIA": 44,
#         "I5": 45,
#         "CEP":46,
#         "PVP":47,
#         "PVQ":48,
#         "DVB":49,
#         "HSN":50,
#         "RME":51,
#         "SIA":52,
#         # "PVPx":53,
#         # "PVPx":54,
#         # "PVPx":55,
#         # "PVPx":56, 
#     }
#     return NEURON_TYPE_ENCODING

def cut_until_r(s: str) -> str:
    """
    Cuts the input string until the first occurrence of lowercase 'r'.
    If 'r' is not found, returns the whole string.
    
    :param s: Input string
    :return: Substring before 'r'
    """
    index = s.find('r')
    return s[:index] if index != -1 else s


if __name__ == "__main__":   

    dataset_root = '/gpfs/radev/home/aa2793/project/cellnn/dataset/Alternative-Splicing'
    expression_file = pd.read_csv(dataset_root+'/main/full_data_Feb01.tsv', delimiter='\t')  
    event_id_list = expression_file['event_id'].tolist() 
    # sample_id_list = expression_file['sample_id'].tolist() 
    # nb_reads_list = expression_file['nb_reads'].tolist() 
    expression_list = expression_file['expression'].tolist()  
    psi_list = expression_file['PSI'].tolist() 
    
     

    old_main_data = pd.read_csv(dataset_root+'/main/2024-12-04-11-35-55_main_data.csv')#, delimiter='\t')  
    event_id_list_old = old_main_data['event_id'].tolist() 
    # neuron_list_old = old_main_data['neuron'].tolist() 
    # nb_reads_list_old = old_main_data['nb_reads'].tolist() 
    p_seq_len_list_old = old_main_data['p_seq_len'].tolist()  
    sj_seq_len_list_old = old_main_data['sj_seq_len'].tolist() 
 
    main_data = {
        "event_id": [],
        "neuron": [],
        "neuron_replicate": [], 
        "nb_reads": [],
        "PSI": [],
        "gene_id": [],
        "p_seq_len": [],
        "sj_seq_len": [], 
        "exon_start": [],
        "exon_end": [],
        "intron_start": [],
        "intron_end": [],
        "gene_length":[]
    }
  
    for i, event_i in enumerate(event_id_list): 
        psi_value = expression_file['PSI'][i]
        gene_length = expression_file['gene_length'][i]
         
        if not np.isnan(psi_value) and not np.isnan(gene_length):
            
            event_id = expression_file['event_id'][i]
            neuron_i = expression_file['sample_id'][i]

            main_data["event_id"].append(event_i) 
            main_data["neuron_replicate"].append(neuron_i)
            neuron_i = cut_until_r(neuron_i)  
            main_data["neuron"].append(neuron_i) 
            nb_reads_i = expression_file['nb_reads'][i]
            main_data["nb_reads"].append(nb_reads_i) 
            main_data["PSI"].append(psi_value)   
            main_data["gene_id"].append(expression_file['gene_id'][i])


            old_main_data_neuron = old_main_data[old_main_data['neuron']==neuron_i] 
            old_main_data_neuron_event = old_main_data_neuron[old_main_data_neuron['event_id']==event_id]
            main_data["p_seq_len"].append(old_main_data_neuron_event['p_seq_len'].item())
            main_data["sj_seq_len"].append(old_main_data_neuron_event['sj_seq_len'].item())
            main_data["exon_start"].append(int(expression_file['exon_start'][i]))
            main_data["exon_end"].append(int(expression_file['exon_end'][i]))   
            main_data["intron_start"].append(int(expression_file['intron_start'][i])) 
            main_data["intron_end"].append(int(expression_file['intron_end'][i])) 
            main_data["gene_length"].append(int(gene_length))
 
            if str(old_main_data_neuron_event['gene_id'].item()) != str(expression_file['gene_id'][i]):
                print('---------------ERROR--!')
             
            print(i)
 
    df = pd.DataFrame(main_data) 
    csv_filename = dataset_root + "/main/2025-02-01-13-40-05_main_data.csv"
    df.to_csv(csv_filename, index=False)

    import pdb;pdb.set_trace() 
    main_data = pd.read_csv(csv_filename)



 














 