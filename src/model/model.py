import torch
from torch import nn 
import pandas as pd  
import torch.nn.functional as F

from data.dataloader import neuron_type_fn 
from einops import rearrange
from model import ztransformer 

import torch


class cellsplicenet(nn.Module):
    def __init__(self, hparams):
        super(cellsplicenet, self).__init__()
        
        # Initialize parameters from hparams 
        self.initialize_parameters(hparams) 

        if (
            hparams.no_sequence_for_ablation or 
            hparams.no_structure_for_ablation or 
            hparams.no_roi_for_ablation or 
            hparams.no_expression_for_ablation
            ): 
            from model import ztransformer_ablation as ztransformer
        else:
            from model import ztransformer 

        self.psi_transformer_encoder = ztransformer.BioZorro(hparams)  
        self.psi_comb_nn = self.psi_build_combining_network()  
        self.psi_pred_nn = self.psi_build_prediction_network()   

        # Experimental control attributes
        self.exp_attn_mat = None
        self.exp_y_vals = None 

    def initialize_parameters(self, hparams):
        """Set class attributes based on hyperparameters."""
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim
        self.embedding_dim = hparams.embedding_dim
        self.ent_weight = hparams.ent_weight
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nb_norm = getattr(hparams, 'nb_norm', True)  
        self.alpha_val = hparams.alpha
        self.beta_val = hparams.beta
        self.gamma_val = hparams.gamma
        self.delta_val = hparams.delta
        self.dev_bool = hparams.dev
        self.seq_scale = hparams.seq_scale
        self.exp_scale = hparams.exp_scale
        self.transform_type = hparams.transform_type
        self.shift_targets = hparams.shift_targets
        self.scale_value = hparams.scale_value
        self.psi_lossfn = hparams.psi_lossfn   
        self.local_seq_length=hparams.local_seq_length 
        self.aux_neuron_loss = hparams.aux_neuron_loss
        self.adj_factor = 1e-2 
        self.fusion_residual = hparams.fusion_residual
        self.device = hparams.device
        self.num_splice_factors = 244
        
 
        self.min_delta_psi = -0.9071368157027444
        self.max_delta_psi = 0.9162412386278128
 
    def psi_build_combining_network(self):
        """Define the combining neural network architecture."""
        layers = [ 
            nn.Linear(self.hidden_dim * 5, self.hidden_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim)
        ]
        return nn.Sequential(*layers)

    def psi_build_prediction_network(self):
        """Define the prediction neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.LogSigmoid()
        )  
 
    def forward(self, data, metadata, embed_list_return=False): 
          
        rna_seq_tensor, rna_annot_tensor = data    
        [output_attention_pooling, expression_tensor, attn_out_data, output_embeds] = self.psi_transformer_encoder(
            metadata, 
            rna_seq_tensor, 
            rna_annot_tensor,
            embed_list_return, 
        )
        
        [
            z_rna_tokens_pool, 
            z_roi_tokens_pool, 
            z_stc_tokens_pool, 
            z_exp_tokens_pool, 
            z_residual_tokens_pool,
            
        ] = output_attention_pooling
            
        transformer_zrep = torch.cat([
            z_rna_tokens_pool, 
            z_roi_tokens_pool, 
            z_stc_tokens_pool, 
            z_exp_tokens_pool, 
            z_residual_tokens_pool
            ],
            dim=1,
        )  
        
        comb_psi_zrep = self.psi_comb_nn(transformer_zrep)    
        expression_tensor = rearrange(expression_tensor, 'b sf d -> b (sf d)')   
        psi_pred_nn = self.psi_pred_nn(comb_psi_zrep)  

        if embed_list_return:
            [z_rna_tokens, z_roi_tokens, z_expression_tokens, fusion_tokens] = output_embeds

            structure_annotation = attn_out_data['structure_info']['structure_annotation']
            structure_seq = attn_out_data['structure_info']['structure_seq']

            embed_list = {
                'z_rna_tokens_pool': z_rna_tokens_pool.cpu(),
                'z_roi_tokens_pool': z_roi_tokens_pool.cpu(),
                'z_structure_tokens_pool': z_stc_tokens_pool.cpu(),
                'z_expression_tokens_pool': z_exp_tokens_pool.cpu(), 
                'z_residual_tokens_pool':z_residual_tokens_pool.cpu(),
                'z_rna_tokens': z_rna_tokens.cpu(),
                'z_roi_tokens': z_roi_tokens.cpu(),
                'z_expression_tokens': z_expression_tokens.cpu(),
                'fusion_tokens': fusion_tokens.cpu(),  
                'transformer_zrep': transformer_zrep.cpu(),
                'rna_attn': attn_out_data['rna_attn'],
                'roi_attn': attn_out_data['roi_attn'],
                'roi_sequence': attn_out_data['roi_sequence'],
                'structurte_attn': attn_out_data['structurte_attn'],
                'experssion_attn_post_transformer': attn_out_data['experssion_attn_post_transformer'], 
                'token_transformer_attention': attn_out_data['token_transformer_attention'], 
                'comb_zrep': comb_psi_zrep.cpu(), 
                # 'comb_deltapsi_zrep': comb_deltapsi_zrep.cpu(), 
                'structure_annotation': structure_annotation,
                'structure_seq': structure_seq,
            }
        else:
            embed_list = None

        perds = { 
            'psi': psi_pred_nn
        } 
         
        return perds, embed_list 

    def loss(self, predictions, targets):  
        
        psi_targets = self.transform_targets(
            targets,  
        )  

        psi_preds = predictions['psi']
 
        # main psi regression loss 
        if self.psi_lossfn == "mse":
            psi_loss = nn.MSELoss(reduction="mean")(psi_preds, psi_targets) 
        elif self.psi_lossfn == "l1":
            psi_loss = nn.L1Loss(reduction="none")(psi_hat, psi_targets)
  
        return psi_loss   
  
    def transform_targets(self, targets): 
        psi_targets = targets['psi'].to(self.device)
        psi_targets = torch.log(psi_targets + self.adj_factor) - self.adj_factor    
        return psi_targets 

    def untransform_targets(self, predictions):   
        psi_preds = torch.exp(predictions['psi'] + self.adj_factor) - self.adj_factor  
        return psi_preds


class deltacellsplicenet(nn.Module):
    def __init__(self, hparams):
        super(deltacellsplicenet, self).__init__()
        
        # Initialize parameters from hparams 
        self.initialize_parameters(hparams) 

        # if (
        #     hparams.no_sequence_for_ablation or 
        #     hparams.no_structure_for_ablation or 
        #     hparams.no_roi_for_ablation or 
        #     hparams.no_expression_for_ablation
        #     ): 
        #     from model import ztransformer_ablation as ztransformer
        # else:
        from model import ztransformer 

        self.psi_transformer_encoder = ztransformer.BioZorro(hparams)  
        self.psi_comb_nn = self.psi_build_combining_network()  
        self.psi_pred_nn = self.psi_build_prediction_network()  
        if self.residual_expresion:
            self.exp_residual = self.res_expression_build_combining_network() 

        # self.delta_psi_comb_nn = self.delta_psi_build_combining_network() 
        # self.delta_psi_pred_nn = self.delta_psi_build_prediction_network() 

        # Experimental control attributes
        self.exp_attn_mat = None
        self.exp_y_vals = None 

    def initialize_parameters(self, hparams):
        """Set class attributes based on hyperparameters."""
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim
        self.embedding_dim = hparams.embedding_dim
        self.ent_weight = hparams.ent_weight
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nb_norm = getattr(hparams, 'nb_norm', True)  
        self.alpha_val = hparams.alpha
        self.beta_val = hparams.beta
        self.gamma_val = hparams.gamma
        self.delta_val = hparams.delta
        self.dev_bool = hparams.dev
        self.seq_scale = hparams.seq_scale
        self.exp_scale = hparams.exp_scale
        self.transform_type = hparams.transform_type
        self.shift_targets = hparams.shift_targets
        self.scale_value = hparams.scale_value
        self.psi_lossfn = hparams.psi_lossfn   
        self.local_seq_length=hparams.local_seq_length 
        self.aux_neuron_loss = hparams.aux_neuron_loss
        self.adj_factor = 1e-2 
        self.fusion_residual = hparams.fusion_residual
        self.device = hparams.device
        self.num_splice_factors = 244
        self.sf_dim = 243
        self.residual_expresion = hparams.residual_expresion
 
        self.min_delta_psi = -0.9071368157027444
        self.max_delta_psi = 0.9162412386278128
 
    def psi_build_combining_network(self):
        """Define the combining neural network architecture."""
        layers = [ 
            nn.Linear(self.hidden_dim * 5, self.hidden_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim)
        ]
        return nn.Sequential(*layers)

    def psi_build_prediction_network(self):
        """Define the prediction neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.LogSigmoid()
        ) 

    def res_expression_build_combining_network(self):
        """Define the combining neural network architecture."""
        layers = [ 
            nn.Linear(self.sf_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim)
        ]
        return nn.Sequential(*layers)
 
    def forward(self, data, metadata, embed_list_return=False): 
          
        rna_seq_tensor, rna_annot_tensor = data    
        [output_attention_pooling, expression_tensor, attn_out_data, output_embeds] = self.psi_transformer_encoder(
            metadata, 
            rna_seq_tensor, 
            rna_annot_tensor,
            embed_list_return, 
        )
        
        [
            z_rna_tokens_pool, 
            z_roi_tokens_pool, 
            z_stc_tokens_pool, 
            z_exp_tokens_pool, 
            z_residual_tokens_pool,
            
        ] = output_attention_pooling
            
        transformer_zrep = torch.cat([
            z_rna_tokens_pool, 
            z_roi_tokens_pool, 
            z_stc_tokens_pool, 
            z_exp_tokens_pool, 
            z_residual_tokens_pool
            ],
            dim=1,
        )  
         
        comb_psi_zrep = self.psi_comb_nn(transformer_zrep)  
        expression_tensor = rearrange(expression_tensor, 'b sf d -> b (sf d)')     
        
         
        if self.residual_expresion:
            expression_tensor_res = self.exp_residual(expression_tensor)  
            psi_pred_nn = self.psi_pred_nn(comb_psi_zrep+expression_tensor_res) 
        else:
            psi_pred_nn = self.psi_pred_nn(comb_psi_zrep) 
      
        if embed_list_return:
            [z_rna_tokens, z_roi_tokens, z_expression_tokens, fusion_tokens] = output_embeds

            structure_annotation = attn_out_data['structure_info']['structure_annotation']
            structure_seq = attn_out_data['structure_info']['structure_seq']
            secondary_structure = attn_out_data['structure_info']['secondary_structure']
            structure_seq_letters = attn_out_data['structure_info']['structure_seq_letters']
            embed_list = {
                'z_rna_tokens_pool': z_rna_tokens_pool.cpu(),
                'z_roi_tokens_pool': z_roi_tokens_pool.cpu(),
                'z_structure_tokens_pool': z_stc_tokens_pool.cpu(),
                'z_expression_tokens_pool': z_exp_tokens_pool.cpu(), 
                'z_residual_tokens_pool':z_residual_tokens_pool.cpu(),
                'z_rna_tokens': z_rna_tokens.cpu(),
                'z_roi_tokens': z_roi_tokens.cpu(),
                'z_expression_tokens': z_expression_tokens.cpu(),
                'fusion_tokens': fusion_tokens.cpu(),  
                'transformer_zrep': transformer_zrep.cpu(),
                'roi_sequence': attn_out_data['roi_sequence'],
                'rna_attn': attn_out_data['rna_attn'],
                'roi_attn': attn_out_data['roi_attn'],
                'structurte_attn': attn_out_data['structurte_attn'],
                'experssion_attn_post_transformer': attn_out_data['experssion_attn_post_transformer'], 
                'fusion_transformer_attention': attn_out_data['fusion_transformer_attention'], 
                'transformer_attention': attn_out_data['transformer_attention'],  
                'comb_zrep': comb_psi_zrep.cpu(), 
                'secondary_structure': secondary_structure,
                # 'comb_deltapsi_zrep': comb_deltapsi_zrep.cpu(), 
                'structure_annotation': structure_annotation,
                'structure_seq': structure_seq,
                'structure_seq_letters': structure_seq_letters
            }
        else:
            embed_list = None

        perds = { 
            'psi': psi_pred_nn
        } 
         
        return perds, embed_list 

    def loss(self, predictions, targets):  
        
        psi_targets = self.transform_targets(
            targets,  
        )  

        psi_preds = predictions['psi']
 
        # main psi regression loss 
        if self.psi_lossfn == "mse":
            psi_loss = nn.MSELoss(reduction="mean")(psi_preds, psi_targets) 
        elif self.psi_lossfn == "l1":
            psi_loss = nn.L1Loss(reduction="none")(psi_hat, psi_targets)
  
        return psi_loss   
  
    def transform_targets(self, targets): 
        psi_targets = targets['psi'].to(self.device)
        psi_targets = torch.log(psi_targets + self.adj_factor) - self.adj_factor    
        return psi_targets 

    def untransform_targets(self, predictions):   
        psi_preds = torch.exp(predictions['psi'] + self.adj_factor) - self.adj_factor  
        return psi_preds
 