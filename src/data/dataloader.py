import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

class dataloader(Dataset):
    def __init__(self, data_csv, sequence_mod_data, dataset_type): #, gene_embed_dict):#, auxiliary_data):

        self.dataset_type = dataset_type 
        # data files
        self.data_csv = data_csv 
        # sequence
        self.prime_seq_dict = sequence_mod_data["primary_sequence"]
        self.sj_seq_dict = sequence_mod_data["sj_sequence"]
        self.sj_inds_df = sequence_mod_data["sj_inds_df"]
        self.max_prime_seq_len = sequence_mod_data["max_prime_seq_len"]
        self.max_sj_seq_len = sequence_mod_data["max_sj_seq_len"]
        self.pad_indx = sequence_mod_data["pad_indx"] 
        # self.gene_embed_dict = gene_embed_dict 
        self.NEURON_TYPE_ENCODING = neuron_type_fn()

    def setup(self):   
        filtered_df = self.data_csv[self.data_csv["p_seq_len"] < self.max_prime_seq_len]   
        self.filtered_df = filtered_df  
        if 'neuron_replicate' not in self.dataset_type: 
            self.events_coordinates = pd.read_csv('~/project/cellnn/dataset/Alternative-Splicing/01Feb2025_replicate/240910_events_coordinates.tsv', sep='\t')
     
    def __getitem__(self, idx):
        raw_datapoint = self.filtered_df.iloc[idx]  

        neuron_idx = torch.LongTensor([self.NEURON_TYPE_ENCODING[raw_datapoint["neuron"]]]).reshape(1, -1)  
        
        if 'neuron_replicate' in self.dataset_type: 
            meta_data = {
                'gene_id': raw_datapoint["gene_id"],
                'event_id': raw_datapoint["event_id"],
                'nb_reads': raw_datapoint["nb_reads"],
                'neuron': raw_datapoint["neuron"],
                'neuron_replicate': raw_datapoint['neuron_replicate'],
                'p_seq_len': raw_datapoint['p_seq_len'],
                'sj_seq_len': raw_datapoint['sj_seq_len'],
                'exon_start': raw_datapoint['exon_start'],
                'exon_end': raw_datapoint['exon_end'],
                'intron_start': raw_datapoint['intron_start'],
                'intron_end': raw_datapoint['intron_end'],
                'gene_length': raw_datapoint['gene_length'], 
                'neuron_idx': neuron_idx
            }
        else: 
            events_coordinates_i = self.events_coordinates[self.events_coordinates['gene_id']==raw_datapoint['gene_id']]
            events_coordinates_i = events_coordinates_i[events_coordinates_i['event_id'] == raw_datapoint['event_id']] 
            exon_start = events_coordinates_i['exon_start'].item()
            exon_end = events_coordinates_i['exon_end'].item()
            intron_start = events_coordinates_i['intron_start'].item()
            intron_end = events_coordinates_i['intron_end'].item()
            gene_length = events_coordinates_i['gene_length'].item() 
            meta_data = {
                'gene_id': raw_datapoint["gene_id"],
                'event_id': raw_datapoint["event_id"],
                'nb_reads': raw_datapoint["nb_reads"],
                'neuron': raw_datapoint["neuron"], 
                'p_seq_len': raw_datapoint['p_seq_len'],
                'sj_seq_len': raw_datapoint['sj_seq_len'], 
                'gene_length': gene_length, 
                'exon_start': exon_start,
                'exon_end': exon_end,
                'intron_start': intron_start,
                'intron_end': intron_end,
            }
  
        # neuron type  
        # sequence modality  
        p_seq = self.prime_seq_dict[raw_datapoint["gene_id"]]
        sj_data = self.sj_seq_dict[raw_datapoint["event_id"]]

        # construct sequence
        p_seq = p_seq.reshape(1, -1)
        sj_annot_seq = sj_data[1, :].reshape(1, -1)

        # get splice region coordinates
        it_sj_df = self.sj_inds_df[self.sj_inds_df["event_id"] == raw_datapoint["event_id"]]
        assert len(it_sj_df) == 1

        # get starting and ending splice junction indieces
        sj_st_indx = int(it_sj_df["sj_st_indx"].iloc[0])
        sj_end_indx = int(it_sj_df["sj_end_indx"].iloc[0])
        end_indx = len(p_seq.flatten())

        # # append to meta data
        # seq_data.append((sj_st_indx, sj_end_indx))

        # convert intron notation to whole sequence notation
        sj_annot_seq[sj_annot_seq == 1] = 1  # exon = 1 (redundant but clarifies next line)
        sj_annot_seq[sj_annot_seq == 0] = 2  # intron = 2

        # flanking rna
        left_p_seq_pad = sj_st_indx
        right_p_seq_pad = end_indx - sj_end_indx
        padded_p_annot_seq = F.pad(sj_annot_seq, (left_p_seq_pad, right_p_seq_pad), value=3)

        # sequence padding
        p_seq = p_seq.flatten() 
        p_padding_size = self.max_prime_seq_len - len(p_seq.flatten())
        padded_p_seq = F.pad(p_seq, (0, p_padding_size), value=self.pad_indx)
        padded_p_seq = padded_p_seq.reshape(1, -1)

        # annotation padding
        padded_p_annot_seq = padded_p_annot_seq.flatten()
        padded_p_annot_seq = F.pad(padded_p_annot_seq, (0, p_padding_size), value=self.pad_indx)
        padded_p_annot_seq = padded_p_annot_seq.reshape(1, -1) 
        # target 
        psi_val = torch.Tensor([raw_datapoint["PSI"]]).reshape(1, 1)
        
        nb_reads = torch.Tensor([raw_datapoint["nb_reads"]]).reshape(1, 1)  
        
        if 'singlereplicant' in self.dataset_type:
            delta_psi_val = torch.Tensor([raw_datapoint["DELTA PSI"]]).reshape(1, 1)
            targets = {
                'psi': psi_val,
                'delta_psi': delta_psi_val,
            }
        else: 
            targets = {
                'psi': psi_val, 
            }
 
        processed_datapoint = [
            meta_data,
            [
                padded_p_seq,
                padded_p_annot_seq, 
                neuron_idx
            ], 
            targets 
        ] 
        
        return processed_datapoint  

    def __len__(self):
        return len(self.filtered_df)


def neuron_type_fn():
    NEURON_TYPE_ENCODING = {
        "AVH": 0,
        "CAN": 1,
        "VD": 2,
        "AFD": 3,
        "RIM": 4,
        "AWC": 5,
        "AIY": 6,
        "RIC": 7,
        "LUA": 8,
        "AVE": 9,
        "RIS": 10,
        "PVC": 11,
        "ADL": 12,
        "ASK": 13,
        "DD": 14,
        "PVM": 15,
        "ASG": 16,
        "AVA": 17,
        "NSM": 18,
        "AVK": 19,
        "ASER": 20,
        "OLQ": 21,
        "SMD": 22,
        "DA": 23,
        "AVM": 24,
        "DVC": 25,
        "VC": 26,
        "AIN": 27,
        "OLL": 28,
        "ASI": 29,
        "IL1": 30,
        "IL2": 31,
        "AIM": 32,
        "SMB": 33,
        "RMD": 34,
        "BAG": 35,
        "AWB": 36,
        "AVL": 37,
        "AWA": 38,
        "AVG": 39,
        "ASEL": 40,
        "VB": 41,
        "PHA": 42,
        "PVD": 43,
        "RIA": 44,
        "I5": 45,
        "CEP":46,
        "PVP":47,
        "PVQ":48,
        "DVB":49,
        "HSN":50,
        "RME":51,
        "SIA":52,
        # "PVPx":53,
        # "PVPx":54,
        # "PVPx":55,
        # "PVPx":56, 
    }
    return NEURON_TYPE_ENCODING

 