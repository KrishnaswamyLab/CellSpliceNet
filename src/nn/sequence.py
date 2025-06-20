import torch 
from torch import nn 

from einops import rearrange, repeat
from einops.layers.torch import Rearrange 
import math

class EntireSeqModality(nn.Module):
    def __init__(self, window_size, new_seq_length, eff_window_size, vocab_size, annot_vocab_size, hidden_dim, pad_indx):
        super().__init__()
       
        self.new_seq_length = new_seq_length
        self.eff_window_size = eff_window_size
        self.vocab_size = vocab_size
        self.pad_indx = pad_indx
        self.annot_vocab_size = annot_vocab_size 
        self.hidden_dim = hidden_dim
        self.seq_patch_dim = window_size * (self.vocab_size + self.annot_vocab_size)

        self.seq_pos_embedding = nn.Embedding(self.new_seq_length, self.hidden_dim)  
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.seq_patch_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        for layer in self.to_patch_embedding:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight) 
        torch.nn.init.xavier_uniform_(self.seq_pos_embedding.weight)
 
        self.seq_glob_attn_module = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            Rearrange("b l c -> b c l"),
            nn.BatchNorm1d(64),
            nn.GELU(),
            Rearrange("b c l -> b l c"),
            nn.Linear(64, 1, bias=False),
            Rearrange("b l c -> b (c l)"), 
            nn.Softmax(dim=1), 
        )
        
        for layer in self.seq_glob_attn_module:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
       
       
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
        annot_ids = self.get_patches(annot_type_tensor, self.pad_indx)                                           # 
        glob_attn_mask = annot_ids.float().mean(-1) != 0       
        annot_ids = torch.eye(self.annot_vocab_size, device=sequence_tensor.device)[annot_ids]  
        return annot_ids, glob_attn_mask

    def prep_sequence(self, sequence_tensor, annot_type_tensor):
        batch_size, inp_seq_len = sequence_tensor.shape

        # RNA Sequence
        # reshape into patches with ending patch padded
        # pad indx = 4  
        sequence_patches = self.get_patches(sequence_tensor, self.pad_indx)                                      #  
        sequence_patches = torch.eye(self.vocab_size, device=sequence_tensor.device)[sequence_patches]           #  
  
        annot_ids, glob_attn_mask = self.annotation_prep(sequence_tensor, annot_type_tensor)                     #  
        comb_seq_tensor = torch.cat([sequence_patches, annot_ids], dim=-1)                                       #  
        comb_seq_tensor = torch.flatten(comb_seq_tensor, 2, 3)                                                    
        
        sequence_patch_embed = self.to_patch_embedding(comb_seq_tensor)                                           

        # positional information
        eff_seq_len = sequence_patch_embed.shape[1]
        position_ids = torch.arange(eff_seq_len, device=sequence_tensor.device)
        position_ids = repeat(position_ids, "n -> b n", b=batch_size)
        positional_patch_embedding = self.seq_pos_embedding(position_ids)

        # sequenece embedding + positional embedding
        sequence_patch_embed = sequence_patch_embed + positional_patch_embedding

        return sequence_patch_embed, glob_attn_mask 

    def seq_glob_attn_op(self, output_seq_embed, mask=None, output_glob_attn=False):
        
        b_size, exp_len, hidden_dim = output_seq_embed.shape 

        for indx, layer in enumerate(self.seq_glob_attn_module):
            if indx == 0:
                z_seq = layer(output_seq_embed)
            else:
                z_seq = layer(z_seq)
 
        mask = mask != mask  # flip booleans  
        z_seq = z_seq.masked_fill(mask, -1e6)  
        
        z_seq = z_seq.reshape(b_size, 1, exp_len)
        z_rep = torch.bmm(z_seq, output_seq_embed)
        z_rep = z_rep.reshape(b_size, hidden_dim)

        if output_glob_attn:
            return z_rep, z_seq.reshape(b_size, exp_len) 
        else:
            return z_rep
    