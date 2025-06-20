import torch
from torch import nn 
import pandas as pd  
import torch.nn.functional as F

from data.dataloader import neuron_type_fn 
from einops import rearrange
from model import ztransformer 
import numpy as np
 

# Function to apply rotation based on the condition
def rotate_values(psi_values, min_psi, max_psi):  
    normalized_angles = 360 * (psi_values - min_psi) / (max_psi - min_psi)
    
    # Apply rotation
    midpoint = 180  # This is 0.5 equivalent when normalized to 0-360
    rotated_angles = torch.where(normalized_angles < midpoint,
                                 normalized_angles + 180,  # Rotate 180 degrees counterclockwise
                                 normalized_angles - 180)  # Rotate 180 degrees clockwise

    # Map back to the range of the original psi_values by reversing the initial normalization
    rotated_psi = (rotated_angles / 360) * (max_psi - min_psi) + min_psi

    return rotated_psi

# Redefine the reverse_rotate_values function with a simpler approach
def reverse_rotate_values_simple(rotated_psi_values, min_psi, max_psi): 
    normalized_angles = 360 * (rotated_psi_values - min_psi) / (max_psi - min_psi)
    reversed_angles = (normalized_angles + 180) % 360  # Shift by 180 degrees and wrap around

    # Map back to the range of original DELTA PSI by reversing the initial normalization
    reversed_psi = (reversed_angles / 360) * (max_psi - min_psi) + min_psi

    return reversed_psi

 

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

        self.transformer_encoder = ztransformer.BioZorro(hparams) 
        
        self.comb_nn = self.build_combining_network() 
        if self.taget_type == 'psi': 
            self.pred_nn = self.build_prediction_network_psi() 
            self.transform_type = 'log'
            self.reduction_type = 'mean'
        elif self.taget_type == 'delta_psi':
            self.pred_nn = self.build_prediction_network_delta_psi() 
            self.transform_type = 'normalize'
            self.reduction_type = 'sum' 

        # Experimental control attributes
        self.exp_attn_mat = None
        self.exp_y_vals = None
        self.mse_loss = nn.MSELoss() 

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
        self.taget_type = hparams.taget_type
        self.residual_expresion = hparams.residual_expresion
        self.min_delta_psi = -0.905075807514832
        self.max_delta_psi = 0.958708187649968
 
    def build_combining_network(self):
        """Define the combining neural network architecture."""
        layers = [
            # nn.Linear(self.hidden_dim, self.hidden_dim * 2), 
            nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2), 
            nn.ReLU(), 
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
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

    def build_prediction_network_psi(self):
        """Define the prediction neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.LogSigmoid()
        ) 
    def build_prediction_network_delta_psi(self):
        """Define the prediction neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            # nn.LogSigmoid()
        ) 

    def forward(self, data, metadata, embed_list_return=True): 
          
        rna_seq_tensor, rna_annot_tensor = data    
        output_attention_pooling, attn_out_data, output_embeds = self.transformer_encoder(
            metadata, 
            rna_seq_tensor, 
            rna_annot_tensor,
            embed_list_return, 
        )
        
        [z_rna_tokens_pool, z_roi_tokens_pool, z_stc_tokens_pool, 
        z_exp_tokens_pool, z_residual_tokens_pool] = output_attention_pooling
            
        transformer_zrep = torch.cat([
            z_rna_tokens_pool, 
            z_roi_tokens_pool, 
            z_stc_tokens_pool, 
            z_exp_tokens_pool, 
            z_residual_tokens_pool
            ],
            dim=1,
        )  
  
        comb_zrep = self.comb_nn(transformer_zrep)  
        pred_nn = self.pred_nn(comb_zrep)
        
        if embed_list_return:
            [z_rna_tokens, z_roi_tokens, z_expression_tokens, fusion_tokens] = output_embeds

            structure_annotation = attn_out_data['structure_info']['structure_annotation']
            structure_seq = attn_out_data['structure_info']['structure_seq']

            embed_list = {
                'z_rna_tokens_pool': z_rna_tokens_pool.cpu(),
                'z_roi_tokens_pool': z_roi_tokens_pool.cpu(),
                'z_structure_tokens_pool': z_stc_tokens_pool.cpu(),
                'z_expression_tokens_pool': z_exp_tokens_pool.cpu(), 
                'z_residual_tokens_pool':z_residual_tokens_pool,
                'z_rna_tokens': z_rna_tokens.cpu(),
                'z_roi_tokens': z_roi_tokens.cpu(),
                'z_expression_tokens': z_expression_tokens.cpu(),
                'fusion_tokens': fusion_tokens.cpu(),  
                'transformer_zrep': transformer_zrep.cpu(),
                'rna_attn': attn_out_data['rna_attn'],
                'roi_attn': attn_out_data['roi_attn'],
                'structurte_attn': attn_out_data['structurte_attn'],
                'experssion_attn_post_transformer': attn_out_data['experssion_attn_post_transformer'], 
                'token_transformer_attention': attn_out_data['token_transformer_attention'], 
                'comb_zrep': comb_zrep.cpu(), 
                'structure_annotation': structure_annotation,
                'structure_seq': structure_seq,
            }
        else:
            embed_list = None
         
        return pred_nn, embed_list
      
    def loss(self, predictions, targets, valid_step=False):
        """ """   
        # unpack everything
        psi_hat = predictions
        psi_target = targets.to(psi_hat.device)
 
        nb_reads = torch.clamp(targets[1], min=0, max=20)
        nb_reads_norm = nb_reads / 20
 
        exp_attn_mat = self.transformer_encoder.save_output_hook.outputs[-1]
        self.transformer_encoder.save_output_hook.clear()

        # for logging
        self.exp_attn_mat = exp_attn_mat.detach()
        self.exp_y_vals = psi_target

        # for stablity
        exp_attn_mat = exp_attn_mat + 1e-5
        exp_attn_mat_entropy = exp_attn_mat * torch.log(exp_attn_mat)

        # sum across token dimension 
        exp_attn_mat_entropy = -exp_attn_mat_entropy.sum(1).mean()

        # weight loss by self.beta_val (cl_args.beta)
        exp_attn_mat_entropy_loss = self.beta_val * exp_attn_mat_entropy

        # transform PSI target
        psi_target = self.transform_targets(psi_target) 

       
        # main PSI regression loss 
        if self.psi_lossfn == "mse":
            psi_loss = nn.MSELoss(reduction="none")(psi_hat, psi_target)
        elif self.psi_lossfn == "l1":
            psi_loss = nn.L1Loss(reduction="none")(psi_hat, psi_target)

        if self.nb_norm:
            normed_psi_loss = psi_loss * nb_reads_norm
        else:
            normed_psi_loss = psi_loss

        if self.reduction_type == 'sum':
            normed_psi_loss = normed_psi_loss.sum() 
        else:
            normed_psi_loss = normed_psi_loss.mean() 

        # variance loss for event id x neuron type 
        if self.aux_neuron_loss == "var_reg":
            eidvar_hat = predictions[1]
            eidvar_true = targets[2]
            eidvar_size = targets[3] 

            eidvar_loss = self.aux_loss_module.calculate_loss(
                yhat=eidvar_hat, ytrue=eidvar_true, ysize=eidvar_size
            ) 
        elif self.aux_neuron_loss in ["clip", "siglip"]:
            eidvar_loss = self.aux_loss_module.calculate_loss()
            self.aux_loss_module.clear_data()

        elif self.aux_neuron_loss in ["none", "noise"]:
            eidvar_loss = 0.0 
        else:
            raise NotImplementedError
  
        total_loss = self.alpha_val * normed_psi_loss
        total_loss += self.beta_val * exp_attn_mat_entropy_loss
        total_loss += self.gamma_val * eidvar_loss

        return total_loss 

 
    def transform_targets(self, targets): 
        if self.transform_type == 'normalize':  
            targets = rotate_values(targets, min_psi=self.min_delta_psi, max_psi=self.max_delta_psi)  

        if self.transform_type == "log": 
            targets = torch.log(targets + self.adj_factor) - self.adj_factor
        
        return targets

    def untransform_targets(self, predictions):
        if self.transform_type == 'normalize': 
            predictions = reverse_rotate_values_simple(predictions, min_psi=self.min_delta_psi, max_psi=self.max_delta_psi)  

        elif self.transform_type == "log": 
            predictions = torch.exp(predictions + self.adj_factor) - self.adj_factor

        return predictions
 