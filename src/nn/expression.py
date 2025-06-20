import torch 
from torch import nn 
import pandas as pd
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from pdb import set_trace as stop
import math

 
class ExpAttention(nn.Module):
    def __init__(self, n_splice_factors, n_neurons):
        super().__init__()

        self.neuron_dictionary = neuron_type_fn()  
        self.alphas = nn.Parameter(torch.empty([n_neurons, n_splice_factors]))   
        nn.init.kaiming_uniform_(self.alphas, a=math.sqrt(5))    
          
    def forward(self, x, neuron_list):  
        x = rearrange(x, 'b n c s -> b n (c s)')
        index = [self.neuron_dictionary[neuron] for neuron in neuron_list]   
        alphas_att = self.alphas[index, :].softmax(1) 
        attn_output  = x * alphas_att.unsqueeze(-1)  
        return attn_output.sum(1), alphas_att
 

class GraphExpressionModality(nn.Module):
    def __init__(self, exp_dim, coeff_dim, hidden_dim, gene_embed_bool, bin_exp, ntype_feature_bool, expression_data_root, save_output_hook):
        super().__init__()
        
        self.hidden_dim = hidden_dim 
        self.exp_dim = exp_dim 
        self.bin_exp = bin_exp
        self.ntype_feature_bool = ntype_feature_bool
        self.gene_embed_bool = gene_embed_bool
        self.coeff_dim = coeff_dim
        self.expression_data_root = expression_data_root 
        self.save_output_hook = save_output_hook 
        
        dataset_root = '/gpfs/radev/home/aa2793/project/cellnn/dataset/Alternative-Splicing/expression_dataset/pp04_scatter/'
        self.scatter_path = dataset_root   
           
        self.expression_dim = 243*11  
        self.to_expression_embedding = nn.Sequential( 
            nn.Linear(self.expression_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
  
        self.exp_glob_attn_module = nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                Rearrange("b l c -> b c l"),
                nn.BatchNorm1d(64),
                nn.GELU(),
                Rearrange("b c l -> b l c"),
                nn.Linear(64, 1, bias=False),
                Rearrange("b l c -> b (c l)"),
                nn.Softmax(dim=1),
        )  

    def prep_expression(self, metadata):    
        expression_data = []  
        for metadata_i in metadata:       
            nueron_i = metadata_i['neuron']  
         
            scatter_path = '/gpfs/radev/home/aa2793/project/cellnn/dataset/Alternative-Splicing/expression_dataset/pp04_scatter/'
            scatter_path = scatter_path + 'scatter_coeffs_'+nueron_i+'.pt'  
            scatter_of_neuron_i = torch.load(scatter_path, weights_only=False)    
            
              
            scatter_of_neuron_i = rearrange(scatter_of_neuron_i, 'a b c -> a c b').unsqueeze(0)
            expression_data.append(scatter_of_neuron_i)   
             
        expression_data = torch.cat(expression_data, dim=0).to('cuda')      
        expression_data = rearrange(expression_data, 'a b c1 c2 -> a b (c1 c2)') 
        embd_expression = self.to_expression_embedding(expression_data)  
        
        return embd_expression, expression_data.mean(-1).unsqueeze(-1) 
    
    def exp_glob_attn_op(self, output_exp_embed, mask=None, output_glob_attn=False):
        """
        glob_attn_weight: B x S tensor
        mask = B x S boolean (False == padding)
        """
        
        b_size, exp_len, hidden_dim = output_exp_embed.shape 
        for indx, layer in enumerate(self.exp_glob_attn_module): 
            
            if indx == 0:
                h = layer(output_exp_embed) 
            else:
                h = layer(h)
       
        att_h = h.reshape(b_size, 1, exp_len)
        z_rep = torch.bmm(att_h, output_exp_embed)
        z_rep = z_rep.reshape(b_size, hidden_dim)
         
        if output_glob_attn:
            return z_rep, att_h.reshape(b_size, exp_len)
        else:
            return z_rep



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


 