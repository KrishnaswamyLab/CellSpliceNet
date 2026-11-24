import torch 
from torch import nn 

from einops import rearrange, repeat
from einops.layers.torch import Rearrange 
import torch.nn.functional as F

import pickle as pk
import pandas as pd

import pickle
 

def load_structure(sequence_tensor, metadata, structure_data_root, structure_length):  
    with open(structure_data_root, 'rb') as file: 
        data = pickle.load(file) 

    structure_seq_list = data['structure_seq']
    secondary_structure_list = data['secondary_structure']
    structure_scattering_coeffs = data['structure_scattering_coeffs']
    gene_id_list = data['gene_id']
    event_id_list = data['event_id']
    exon_coordinates_list = data['exon_coordinates']
    intron_coordinates_list = data['intron_coordinates'] 
      
    batch_seqstc_coeff = [] 
    structure_annotation = []
    vocab_map = {"PAD": 0, "A": 1, "G": 2, "T": 3, "C": 4, "X": 5}
    # annot_map = {"PAD": 0, "EXON": 1, "INTRON": 2, "FLANK": 3}
    structure_seq_cat = []
    secondary_structure=[]
    structure_seq_letters = []
 
    for i, metadata_i in enumerate(metadata):  
        event_id_i = metadata_i["event_id"] 

        index_of_event_in_expression_data = event_id_list.index(event_id_i)  

        structure_seq_i = structure_seq_list[index_of_event_in_expression_data]
        secondary_structure_i = secondary_structure_list[index_of_event_in_expression_data]
        exon_coordinates_i = exon_coordinates_list[index_of_event_in_expression_data]      
        intron_coordinates_i = intron_coordinates_list[index_of_event_in_expression_data]   
        strcutre_coeff_i = structure_scattering_coeffs[index_of_event_in_expression_data]  

        # check if the coordinates of the main file (batch) and strcutures are same
        exon_coord_i_of_batch = (metadata_i['exon_start'], metadata_i['exon_end'])                   
        intront_coord_i_of_batch = (metadata_i['intron_start'], metadata_i['intron_end'])        
        if (exon_coord_i_of_batch != exon_coordinates_i) or (intront_coord_i_of_batch != intron_coordinates_i):
            raise ValueError("The corrdinates are not equal.")

        # define the strcuture annotation and consider 2 as exon 
        half_exon_coordinate_i = int((exon_coordinates_i[-1]-exon_coordinates_i[0])/2) 
        if half_exon_coordinate_i > structure_length:
            half_exon_coordinate_i = structure_length 
        stc_length_half = int(structure_length/2)
        strcutre_coeff_i = strcutre_coeff_i.sum(-1)

        bp_discrete = torch.zeros(structure_length)
        for seq_index, bp in enumerate(structure_seq_i):   
            bp_discrete[seq_index] = vocab_map[bp]
 
        padding_size =  structure_length - strcutre_coeff_i.size(0)
     
        structure_annotation_i = torch.zeros(strcutre_coeff_i.size(0))+2
        structure_annotation_i[stc_length_half-half_exon_coordinate_i:stc_length_half+half_exon_coordinate_i] -= 1
 
        strcutre_coeff_i = F.pad(strcutre_coeff_i, (0, 0, 0, padding_size), 'constant', 0)  
        structure_annotation_i = F.pad(structure_annotation_i, (0, padding_size), 'constant', 0)  
  
        structure_seq_cat.append(bp_discrete.unsqueeze(0)) 
        batch_seqstc_coeff.append(strcutre_coeff_i.unsqueeze(0))
        structure_annotation.append(structure_annotation_i.unsqueeze(0))
        secondary_structure.append(secondary_structure_i)
        structure_seq_letters.append(structure_seq_i)
 
    structure_seq = torch.cat(structure_seq_cat, dim=0).to('cuda') 
    batch_seqstc_coeff = torch.cat(batch_seqstc_coeff).to('cuda')   
    structure_annotation = torch.cat(structure_annotation).to('cuda')  
    structure_annotation.requires_grad = True 
 
    min_val = torch.min(batch_seqstc_coeff)
    max_val = torch.max(batch_seqstc_coeff)
    batch_seqstc_coeff = (batch_seqstc_coeff - min_val) / (max_val - min_val)
 
    return batch_seqstc_coeff, structure_annotation, structure_seq, secondary_structure, structure_seq_letters

 
class StructureModality(nn.Module):
    def __init__(self, seq_coeff_dim, hidden_dim, structure_data_root, structure_length):
        super().__init__()

        self.structure_data_root = structure_data_root
        self.hidden_dim = hidden_dim 
        self.seq_coeff_dim = hidden_dim  
        self.structure_length = structure_length

        self.to_scattering_embedding = nn.Sequential( 
            nn.Linear(17, self.hidden_dim), 
            nn.LayerNorm(self.hidden_dim),
        )
        
        self.structure_pos_embedding = nn.Embedding(self.structure_length, self.hidden_dim) 
        torch.nn.init.xavier_uniform_(self.structure_pos_embedding.weight)

        self.scattering_coeff_attn_module = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            Rearrange("b l c -> b c l"),
            nn.BatchNorm1d(64),
            nn.GELU(),
            Rearrange("b c l -> b l c"),
            nn.Linear(64, 1, bias=False),
            Rearrange("b l c -> b (c l)"), 
            nn.Softmax(dim=1),
        )

        self.new_stc_length = self.structure_length
        self.eff_window_size = 1  # 64
        self.vocab_size = 6
        self.pad_indx = 0
        self.annot_vocab_size = 6

    def get_patches(self, input_tensor, pad_indx):
        bs_size = input_tensor.shape[0]
        list_of_patches = list(torch.chunk(input_tensor, self.new_stc_length, dim=1))

        # ensure size is (B, self.window_size) for all fragments 
        if len(set([x.shape[1] for x in list_of_patches])) == 1:
            pass
        else:
            pad_to_len = self.eff_window_size - list_of_patches[-1].shape[1]
            padding = (
                torch.Tensor([pad_indx]).repeat(bs_size, pad_to_len).to(input_tensor.device)
            )
            list_of_patches[-1] = torch.cat([list_of_patches[-1], padding], dim=1)

        list_of_patches = [x.unsqueeze(1) for x in list_of_patches]
        patch_tensor = torch.cat(list_of_patches, dim=1).to(torch.long)

        return patch_tensor 

    def annotation_prep(self, annot_type_tensor):
        annot_ids = self.get_patches(annot_type_tensor, self.pad_indx)                                           # [16, 938, 32]   
        glob_attn_mask = annot_ids.float().mean(-1) != 0       
        annot_ids = torch.eye(self.annot_vocab_size, device=annot_type_tensor.device)[annot_ids] 
        return annot_ids, glob_attn_mask
    
    def prep_scattering_coeff(self, sequence_tensor, metadata):     
        
        local_structure_tensor, structure_annotation, structure_seq, secondary_structure, structure_seq_letters = load_structure(sequence_tensor, metadata, self.structure_data_root, self.structure_length)  
        batch_size = local_structure_tensor.shape[0] 
        annot_ids, glob_attn_mask = self.annotation_prep(structure_annotation) 
 
        comb_structure_tensor = torch.cat([local_structure_tensor.unsqueeze(2) , annot_ids], dim=-1)  
        comb_structure_tensor = torch.flatten(comb_structure_tensor, 2, 3) 
 
        local_structure_patch_embed = self.to_scattering_embedding(comb_structure_tensor)     # [16, 938, 512]
 
        coff_len = local_structure_patch_embed.shape[1]
        position_ids = torch.arange(coff_len, device=local_structure_patch_embed.device)
        
        position_ids = repeat(position_ids, "n -> b n", b=batch_size)
        positional_patch_embedding = self.structure_pos_embedding(position_ids) 
        local_structure_patch_embed = local_structure_patch_embed + positional_patch_embedding 
         
        return local_structure_patch_embed, structure_annotation, structure_seq, secondary_structure, structure_seq_letters

    def scattering_coeff_glob_attn_op(self, output_seq_embed, output_glob_attn=False): 
        b_size, exp_len, hidden_dim = output_seq_embed.shape
 
        for indx, layer in enumerate(self.scattering_coeff_attn_module):
            if indx == 0:
                z_seq = layer(output_seq_embed)
            else:
                z_seq = layer(z_seq)
                
        z_seq = z_seq.reshape(b_size, 1, exp_len)
        z_rep = torch.bmm(z_seq, output_seq_embed)
        z_rep = z_rep.reshape(b_size, hidden_dim)

        if output_glob_attn:
            return z_rep, z_seq.reshape(b_size, exp_len) 
        else:
            return z_rep
 