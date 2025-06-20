from argparse import Namespace 
import pandas as pd
import numpy as np 

import torch 
from torch.utils.data import DataLoader
 
import pathlib

from data.dataloader import dataloader  
from utils.config_utils import get_config_val_from_file as config_read

 
class splicedata_dataloader():
    """ """

    def __init__(self, hparams):
        super().__init__()

        self.batch_size = hparams.batch_size
        self.max_prime_seq_len = hparams.max_prime_seq_len
        self.max_sj_seq_len = hparams.max_sj_seq_len
        self.data_size = hparams.data_size
        self.config_fname = hparams.config_fname
        self.no_ci = hparams.no_ci

        self.gene_embed_bool = hparams.gene_embed_bool
        self.bin_exp = hparams.bin_exp
        self.ntype_feature_bool = hparams.ntype_feature_bool
        self.exp_norm_bool = hparams.exp_norm_bool

        self.sfgenes_bool = hparams.sfgenes_bool
        self.sfgenes = hparams.sfgenes

        self.topkgenes_bool = hparams.topkgenes_bool
        self.topkgenes = hparams.topkgenes 
        
        self.data_dir = pathlib.Path(hparams.config_fname).parent
        self.dataset_type = hparams.dataset_type

         
        
        #     from nn.expression_singlereplicant import ExpressionModality as ExpressionModality
        # else: 
        #     from nn.expression import GraphExpressionModality as ExpressionModality
        
        
        # self.events_coordinates = pd.read_csv(hparams.dataset_root+'Alternative-Splicing/main/240910_events_coordinates.tsv', sep='\t')  
        # 
        # file paths are pulled from config file   
        self.enc_prime_seq_fname = config_read(self.config_fname, "processed_files", "enc_seq_file")
        self.enc_sj_seq_fname = config_read(self.config_fname, "processed_files", "enc_sj_file")
        # self.exp_csv_fname = config_read(self.config_fname, "processed_files", "exp_by_nid_t_file")

        # self.sf_tsv_fname = config_read(self.config_fname, "processed_files", "sf_list")
        self.sj_inds_fname = config_read(self.config_fname, "processed_files", "spliceregion_inds")
        # self.var_genes_fname = config_read(self.config_fname, "processed_files", "top300variablegenes")

        # gene embeds - new 
        # self.gene_embed_array_fname = config_read(self.config_fname, "processed_files", "graph_node_embeds")
        # self.gene_embed_dict_fname = config_read(self.config_fname, "processed_files", "graph_node_names")
        # self.gene_embed_names_fname = config_read(self.config_fname, "processed_files", "graph_node_names")
        # self.gene_embed_stds_fname = config_read(self.config_fname, "processed_files", "graph_node_stds")
  
        # -----
        # self.neurontype_features = config_read(self.config_fname, "processed_files", "neurontype_features")
        self.train_fname = config_read(self.config_fname, "processed_files", "train_data_file")
        self.valid_fname = config_read(self.config_fname, "processed_files", "valid_data_file")
        self.test_fname = config_read(self.config_fname, "processed_files", "test_data_file")

        self.vocab_map = {"PAD": 0, "A": 1, "G": 2, "U": 3, "C": 4, "X": 5}
        self.annot_map = {"PAD": 0, "EXON": 1, "INTRON": 2, "FLANK": 3}
 
        self.pad_indx = 0 

    def splice_factor_prep(self, exp_csv, final_geneset, embedded_genes_list):  
         
        # i) filter 243 splice factors (final_geneset) all 44973 of the expression matrix (exp_csv)  
        intersection_genes = set(final_geneset).intersection(set(embedded_genes_list)) 
        splicefactors_neuron_matrix = exp_csv[exp_csv["gene_id"].isin(intersection_genes)]  
        
        # ii)  log transformation
        gene_id = splicefactors_neuron_matrix["gene_id"]
        splicefactors_neuron_matrix = splicefactors_neuron_matrix.drop(["gene_id"], axis=1) 
        splicefactors_neuron_matrix = np.log(splicefactors_neuron_matrix + 1e-2)
        splicefactors_neuron_matrix = pd.concat([gene_id, splicefactors_neuron_matrix], axis=1)  
        
        return splicefactors_neuron_matrix

    def setup(self):   

        enc_prime_seq_dict = torch.load(self.enc_prime_seq_fname, weights_only=True)
        enc_sj_seq_dict = torch.load(self.enc_sj_seq_fname, weights_only=True)
        sj_inds_df = pd.read_csv(self.sj_inds_fname) 

        sequence_modality_data = {
            "primary_sequence": enc_prime_seq_dict,
            "sj_sequence": enc_sj_seq_dict,
            "sj_inds_df": sj_inds_df,
            "max_prime_seq_len": self.max_prime_seq_len,
            "max_sj_seq_len": self.max_sj_seq_len,
            "pad_indx": self.pad_indx,
        }

        # graph_node_stds = ../dataset/processed/addtnl_data/genecoeffs_gstds.csv 
        # EXPRESSION MODALITY: Splice Factors and their Scattering Embeddings
        # load the expression data
        # expression_csv = pd.read_csv(self.exp_csv_fname)                                  # ../dataset/processed_v02/240910/2024-09-11-14-06-12_expression_by_neurontype_t.csv
        # sf_df = pd.read_csv(self.sf_tsv_fname, delimiter="\t")                            # (243, 3) 
        # sf_list = sf_df["gene_id"].tolist()[: self.sfgenes]                              
        
        # load the expression embedded data (coeff)
        # expression_coeff_array = np.load(self.gene_embed_array_fname)                           # shape [44973, 132]  '../dataset/processed_v02/240910/entities_kg_emb_2023-03-10_celegans_KG2E_nepochs_20.npy'

        # self.coeff_dim = expression_coeff_array.shape[-1]

        # read index and name of the coeffs 
        # gene_embed_stds = pd.read_csv(self.gene_embed_stds_fname)                                 # [44973, 3]
        # gene_embed_dict_name_index = {k: i for i, k in enumerate(gene_embed_stds["gene_id"])} 
        
        # # splice factors extraction: filtering sf, and log transform
        # splice_factor_csv = self.splice_factor_prep(
        #     expression_csv, sf_list, list(gene_embed_dict_name_index.keys())
        # )
         
        # filtering the splice factor embeddings (coeff)
        # gene_embed_inds = [gene_embed_dict_name_index[x] for x in splice_factor_csv["gene_id"]]             
        # expression_coeff_array = expression_coeff_array[gene_embed_inds]                               ### off after new jake dataset
        
        # Neuron Features
        # neuron_features_df = pd.read_csv(
        #     self.neurontype_features, delimiter="\t"
        # )
          
        # expression_modality_data = {
        #     "expression": splice_factor_csv, 
        #     "sf_list": sf_list, 
        #     # "exp_dim": splice_factor_csv.shape[0],
        #     # "expression_coeff_array": expression_coeff_array,
        #     "neuron_features_df": neuron_features_df,
        # }
         
        # aux_data = {
        #     "gene_embed_bool": self.gene_embed_bool,
        #     "gene_embed_dict": gene_embed_dict_name_index,   
        #     "ntype_feature_bool": self.ntype_feature_bool, 
        #     "data_size": self.data_size,
        # }

        # DATAPOINT  
        train_csv = pd.read_csv(self.train_fname)
        train_data = dataloader(
            data_csv=train_csv,
            sequence_mod_data=sequence_modality_data,
            dataset_type=self.dataset_type,
            # gene_embed_dict = gene_embed_dict_name_index
            # expression_mod_data=expression_modality_data,
            # auxiliary_data=aux_data,
        )
        train_data.setup()
 
        valid_csv = pd.read_csv(self.valid_fname)
        valid_data = dataloader(
            data_csv=valid_csv,
            sequence_mod_data=sequence_modality_data,
            dataset_type=self.dataset_type,
            # gene_embed_dict = gene_embed_dict_name_index
            # expression_mod_data=expression_modality_data,
            # auxiliary_data=aux_data,
        )
        valid_data.setup()
 
        test_csv = pd.read_csv(self.test_fname) 
        test_data = dataloader(
            data_csv=test_csv,
            sequence_mod_data=sequence_modality_data,
            dataset_type=self.dataset_type,
            # gene_embed_dict = gene_embed_dict_name_index
            # expression_mod_data=expression_modality_data,
            # auxiliary_data=aux_data,
        )
        test_data.setup()

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.train_N = len(train_data)
        self.valid_N = len(valid_data)
        self.test_N = len(test_data) 

        # self.exp_dim = train_data.exp_data_csv.shape[0]
        # self.gene_embed_dim = expression_coeff_array.shape[-1]
        # self.ntype_feature_dim = neuron_features_df.shape[-1] - 1
 
    def train_dataloader(self, shuffle_bool=True):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=shuffle_bool,
            collate_fn=sp3_collate_fn,
            drop_last=True
        )

    def valid_dataloader(self, shuffle_bool=True):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=sp3_collate_fn,
            drop_last=True
        )

    def test_dataloader(self, shuffle_bool=True):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=sp3_collate_fn,
            drop_last=True
        )
        
    def setup_hparams(self, hparams):
        """
        update hparam.datahparams dictionary with
        dataset specific values
        """

        datahparam_dict = {}

        # for seq model
        datahparam_dict["max_prime_seq_len"] = self.max_prime_seq_len
        datahparam_dict["max_sj_seq_len"] = self.max_sj_seq_len
        # datahparam_dict["coeff_dim"] = self.coeff_dim
        # datahparam_dict["exp_dim"] = self.exp_dim
        # datahparam_dict["gene_embed_dim"] = self.gene_embed_dim
        # datahparam_dict["ntype_feature_dim"] = self.ntype_feature_dim

        datahparam_dict["vocab_size"] = len(self.vocab_map)
        datahparam_dict["vocab_map"] = self.vocab_map
        datahparam_dict["annot_map"] = self.annot_map

        datahparam_dict["pad_indx"] = self.pad_indx
        # general
        datahparam_dict["batch_size"] = self.batch_size
        datahparam_dict["dataset"] = self.__class__.__name__

        hparams.dataparams = Namespace(**datahparam_dict) 
        return hparams
     

def sp3_collate_fn(batch):  
    
    seq_data, data_x, data_y = zip(*batch)  

    (
        padded_prime_seq,
        padded_p_annot_seq,
        neuron_type,
        # neuron_name,
        # neuron_replicate,
        # gene_id,
        # p_seq_len,
        # sj_seq_len,
        # exon_start,
        # exon_end,
        # intron_start,
        # intron_end,
        # gene_length
    ) = zip(*data_x) 
    
    rna_seq_tensor = torch.cat(padded_prime_seq)
    annot_seq_tensor = torch.cat(padded_p_annot_seq)
    neuron_type_tensor = torch.cat(neuron_type)
    # neuron_feat_tensor = torch.cat(neuron_feat)
    # exp_tensor = torch.cat(exp_tensor)
    # gene_embed_tensor = torch.cat(gene_embeds)
    # exp_gene_embed_tensor = torch.cat(exp_gene_embeds)
  
    psi_y_tensor = torch.cat([x['psi'] for x in data_y])  

    
    # delta_psi_y_tensor = torch.cat([x['delta_psi'] for x in data_y])
    delta_psi_y_tensor = [x['delta_psi'] if 'delta_psi' in x else None for x in data_y]
    targets = {
        'psi': psi_y_tensor,
        'delta_psi': delta_psi_y_tensor,
    }
    


    # eidvar_tensor = torch.cat([x[2] for x in data_y])
    # eidsize_tensor = torch.cat([x[3] for x in data_y]) 
    
    # ### NEED TO LOAD ONCE
    # dataset_root = '/gpfs/radev/project/lafferty/aa2793/splicenn/dataset/'
    # events_coordinates = pd.read_csv(dataset_root+'/processed_v02/main/240910_events_coordinates.tsv', sep='\t')  
    # # import pdb;pdb.set_trace()
    # local_seq_tensor = local_seq_extraction_no_batch(rna_seq_tensor, events_coordinates, meta_data, local_seq_length=200)
    
    # new_batch = [
    #     meta_data,
    #     [
    #         rna_seq_tensor,
    #         annot_seq_tensor, 
    #         neuron_type_tensor, 
    #         neuron_name,
    #         neuron_replicate,
    #         gene_id,
    #         p_seq_len,
    #         sj_seq_len,
    #         exon_start,
    #         exon_end,
    #         intron_start,
    #         intron_end,
    #         gene_length
    #     ],
    #     [psi_y_tensor], #, eidvar_tensor, eidsize_tensor],
    # ]

    

    new_batch = [
        seq_data,
        [ 
            rna_seq_tensor,
            annot_seq_tensor,
            # local_seq_tensor,
            # neuron_type_tensor,
            # neuron_name,
            # neuron_replicate,
            # gene_id,
            # p_seq_len,
            # sj_seq_len,
            # exon_start,
            # exon_end,
            # intron_start,
            # intron_end,
            # gene_length
        ],
        targets, #, eidvar_tensor, eidsize_tensor],
    ] 
    
    return new_batch

 


















def local_seq_extraction(rna_seq_tensor, metadata, events_coordinates, local_seq_length = 200):  
    batch_size = len(metadata)
    local_seq_data = []
    for example_index in range(batch_size):   
        geneid_i, eventid_i, _, neuron_i, sj_coords = metadata[example_index]  
        seq_data = rna_seq_tensor[example_index, :]   
        interested_gene_coordinates = events_coordinates[events_coordinates['gene_id']==geneid_i]
        interested_gene_coordinates = interested_gene_coordinates[interested_gene_coordinates['event_id']==eventid_i]
        exon_start = interested_gene_coordinates['exon_start'].item()  
        exon_end = interested_gene_coordinates['exon_end'].item()  
         
        border_length = int(local_seq_length/4) 
        
        # cut the sequence - border_length to exon_start & +border_length after exon_end
        cut_start_point = exon_start-border_length
        cut_end_point = exon_end+border_length
        
        # check if exon is less than the begining of sequence then set it  exon_start
        if cut_start_point < 0:
            cut_start_point = exon_start 
            
        # check if cut_end_point is greater than the end of sequence then set it to exon_end
        if cut_end_point > len(seq_data):
            cut_end_point = exon_end 
        
        # cut the exon -+ border_length as local sequence  
        local_seq_data_i = seq_data[cut_start_point:cut_end_point]  
        
        # check if  exon -+ border_length is greather than local_seq_length then drop the middle of the exon
        if len(local_seq_data_i) > local_seq_length:  
            left_local_seq_data_i = local_seq_data_i[:border_length*2]
            right_local_seq_data_i = local_seq_data_i[len(local_seq_data_i)-(border_length*2):]
            local_seq_data_i = torch.cat((left_local_seq_data_i, right_local_seq_data_i), dim=0)
             
        # check if  exon -+ border_length is less than local_seq_length then drop the middle of the exon
        pad_right = local_seq_length - local_seq_data_i.size(0) 
        if pad_right>0:
            pad_left = 0
            pad_right = pad_right
            padding = (pad_left, pad_right)
            local_seq_data_i = torch.nn.functional.pad(local_seq_data_i, padding, mode='constant', value=0)  
        
        local_seq_data.append(local_seq_data_i.unsqueeze(0).float())
        
    local_seq_data = torch.cat(local_seq_data, dim=0)
    return local_seq_data




 