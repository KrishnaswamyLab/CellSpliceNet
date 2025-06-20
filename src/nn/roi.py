import torch 
from torch import nn 

from einops import rearrange, repeat
from einops.layers.torch import Rearrange 
import pandas as pd


def local_seq_extraction(rna_seq_tensor, annot_seq_tensor, metadata, local_seq_length = 200):  
    local_seq_data = []
    local_annot_data = []
    for i, metadata_i in enumerate(metadata):   
        seq_data = rna_seq_tensor[i, :]  
        annot_seq = annot_seq_tensor[i, :] 
        exon_start = metadata_i['exon_start']
        exon_end = metadata_i['exon_end']
         
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
        local_annot_data_i = annot_seq[cut_start_point:cut_end_point]  
        
        # check if  exon -+ border_length is greather than local_seq_length then drop the middle of the exon
        if len(local_seq_data_i) > local_seq_length:  
            left_local_seq_data_i = local_seq_data_i[:border_length*2]
            right_local_seq_data_i = local_seq_data_i[len(local_seq_data_i)-(border_length*2):]
            local_seq_data_i = torch.cat((left_local_seq_data_i, right_local_seq_data_i), dim=0)

            left_local_annot_data_i = local_annot_data_i[:border_length*2]
            right_local_annot_data_i = local_annot_data_i[len(local_annot_data_i)-(border_length*2):]
            local_annot_data_i = torch.cat((left_local_annot_data_i, right_local_annot_data_i), dim=0)
             
        # check if  exon -+ border_length is less than local_seq_length 
        # then drop the middle of the exon
        pad_right = local_seq_length - local_seq_data_i.size(0) 

        if pad_right > 0:
            pad_left = 0
            pad_right = pad_right
            padding = (pad_left, pad_right)
            local_seq_data_i = torch.nn.functional.pad(local_seq_data_i, padding, mode='constant', value=0)  
            local_annot_data_i = torch.nn.functional.pad(local_annot_data_i, padding, mode='constant', value=0)  
        
        local_seq_data.append(local_seq_data_i.unsqueeze(0).float())
        local_annot_data.append(local_annot_data_i.unsqueeze(0).float())
  
    local_seq_data = torch.cat(local_seq_data, dim=0)
    local_annot_data = torch.cat(local_annot_data, dim=0)
     
    return local_seq_data, local_annot_data 

 
class LocalSeqModality(nn.Module):
    def __init__(self, hparams, hidden_dim):
        super().__init__()
         
        self.hidden_dim = hidden_dim 
        self.to_local_embedding = nn.Sequential( 
                nn.Linear(10, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            ) 
        for layer in self.to_local_embedding:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight) 

        new_seq_length = 200
        self.roi_pos_embedding = nn.Embedding(new_seq_length, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.roi_pos_embedding.weight)
          
        self.local_seq_glob_attn_module = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            Rearrange("b l c -> b c l"),
            nn.BatchNorm1d(64),
            nn.GELU(),
            Rearrange("b c l -> b l c"),
            nn.Linear(64, 1, bias=False),
            Rearrange("b l c -> b (c l)"), 
            nn.Softmax(dim=1),
        )
        for layer in self.local_seq_glob_attn_module:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        self.new_seq_length = new_seq_length
        self.eff_window_size = 1  # 64
        self.vocab_size = 6
        self.pad_indx = 0
        self.annot_vocab_size = 4

    def get_patches(self, input_tensor, pad_indx):
        bs_size = input_tensor.shape[0]
        list_of_patches = list(torch.chunk(input_tensor, self.new_seq_length, dim=1))

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

    def annotation_prep(self, sequence_tensor, annot_type_tensor):
        annot_ids = self.get_patches(annot_type_tensor, self.pad_indx)                                           # [16, 938, 32]   
        glob_attn_mask = annot_ids.float().mean(-1) != 0       
        annot_ids = torch.eye(self.annot_vocab_size, device=sequence_tensor.device)[annot_ids]  
        return annot_ids, glob_attn_mask
 
    
    def prep_local_sequence(self, metadata, sequence_tensor, seq_annotation_tensor, local_seq_length): 
        
        local_sequence_tensor, local_annotation_tensor = local_seq_extraction(
            sequence_tensor, seq_annotation_tensor, metadata, 
            local_seq_length=local_seq_length
        ) 

        batch_size = local_sequence_tensor.shape[0] 
        
        local_sequence_tensor_patch = self.get_patches(local_sequence_tensor, self.pad_indx) 
        local_sequence_tensor_patch = torch.eye(self.vocab_size, device=sequence_tensor.device)[local_sequence_tensor_patch]           # [16, 938, 32, 6]
 
        annot_ids, glob_attn_mask = self.annotation_prep(local_sequence_tensor_patch, local_annotation_tensor)
        comb_local_seq_tensor = torch.cat([local_sequence_tensor_patch, annot_ids], dim=-1)             
        
        comb_local_seq_tensor = torch.flatten(comb_local_seq_tensor, 2, 3) 
        local_sequence_patch_embed = self.to_local_embedding(comb_local_seq_tensor)                                          # [16, 938, 512]

        # positional information
        eff_seq_len = local_sequence_patch_embed.shape[1]
        position_ids = torch.arange(eff_seq_len, device=sequence_tensor.device)
        position_ids = repeat(position_ids, "n -> b n", b=batch_size)
        positional_patch_embedding = self.roi_pos_embedding(position_ids)

        # sequenece embedding + positional embedding 
        local_sequence_patch_embed = local_sequence_patch_embed + positional_patch_embedding 

        return local_sequence_patch_embed, local_sequence_tensor 

    def local_seq_glob_attn_op(self, output_seq_embed, output_glob_attn=False):
        
        b_size, exp_len, hidden_dim = output_seq_embed.shape 

        for indx, layer in enumerate(self.local_seq_glob_attn_module):
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

 