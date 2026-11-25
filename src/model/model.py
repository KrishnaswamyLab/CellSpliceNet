import torch
from torch import nn 
import pandas as pd  
import torch.nn.functional as F

from data.dataloader import neuron_type_fn 
from einops import rearrange
from model import ztransformer

import torch
import os


# ---------------------------------------------------------------------------
# Opt-in PSI output heads for deltacellsplicenet (and every variant built on it),
# selected with the env var PSI_HEAD at construction time:
#   log     (default) original behaviour: LogSigmoid head, MSE on log(PSI + 0.01)
#   linear  sigmoid head, MSE on raw PSI (no log transform)
#   twohead exon head  = mean PSI of the exon (event) across tissues/neurons
#           dev  head  = PSI - exon mean (tissue/neuron-specific deviation)
#           loss = MSE(exon)/var(exon means) + MSE(dev)/var(deviations), both on raw PSI
#           prediction = clamp(exon + dev, 0, 1)
# Exon-mean targets are the mean PSI of that (event) over the rows of the same data file
# (train / valid / test files of --config_fname, via utils.paths); (event_id, neuron) pairs are unique across
# the files in every split we use. The lookup lives in a module cache (not in the
# checkpoint) and is rebuilt lazily from the config path after torch.load.
# ---------------------------------------------------------------------------
_EVENT_MEAN_CACHE = {}


def _event_mean_lookup(config_fname):
    if config_fname not in _EVENT_MEAN_CACHE:
        import configparser
        from pathlib import Path
        from utils.paths import FILES_SECTION, resolve
        cfg = configparser.ConfigParser(); cfg.read(config_fname)
        # [files] (current format); [processed_files] kept so checkpoints trained before the
        # paths.py refactor still load
        sec = cfg[FILES_SECTION] if FILES_SECTION in cfg else cfg["processed_files"]
        lut, stats = {}, None
        for key in ("train_data_file", "valid_data_file", "test_data_file"):
            if key not in sec:
                continue
            f = resolve(sec[key], config_file=Path(config_fname))
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f, usecols=["event_id", "neuron", "PSI"])
            d = d.groupby(["event_id", "neuron"], as_index=False).PSI.mean()
            d["m"] = d.groupby("event_id").PSI.transform("mean")
            lut.update(zip(d.event_id.astype(str) + "|" + d.neuron.astype(str), d.m.astype(float)))
            if key == "train_data_file":
                stats = (float(d.m.var()), float((d.PSI - d.m).var()))
        _EVENT_MEAN_CACHE[config_fname] = (lut, stats)
        print(f"[psi_head] event-mean lookup: {len(lut)} (event, neuron) rows; "
              f"train var(exon mean)={stats[0]:.4f} var(dev)={stats[1]:.5f}", flush=True)
    return _EVENT_MEAN_CACHE[config_fname]


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
        self.num_splice_factors = hparams.sfgenes + 1

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
            psi_loss = nn.L1Loss(reduction="mean")(psi_preds, psi_targets)
  
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

        # opt-in output heads (see module header); default "log" leaves the model unchanged
        self.psi_head = os.environ.get("PSI_HEAD", "log")
        if self.psi_head not in ("log", "linear", "twohead"):
            raise ValueError(f"unknown PSI_HEAD={self.psi_head}")
        if self.psi_head != "log":
            self.psi_pred_nn = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.Sigmoid())
        if self.psi_head == "twohead":
            self.dev_pred_nn = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, 1), nn.Tanh())
            self.event_mean_cfg = hparams.config_fname
            _, (self.var_exon, self.var_dev) = _event_mean_lookup(self.event_mean_cfg)
        print(f"[psi_head] {type(self).__name__}: PSI_HEAD={self.psi_head}", flush=True)

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
        self.num_splice_factors = hparams.sfgenes + 1
        self.sf_dim = hparams.sfgenes
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
            z_head = comb_psi_zrep + expression_tensor_res
        else:
            z_head = comb_psi_zrep
        psi_pred_nn = self.psi_pred_nn(z_head)
        psi_head = getattr(self, "psi_head", "log")
        if psi_head == "twohead":
            exon_pred = psi_pred_nn
            dev_pred = self.dev_pred_nn(z_head)
            psi_pred_nn = (exon_pred + dev_pred).clamp(0.0, 1.0)
            lut, _ = _event_mean_lookup(self.event_mean_cfg)
            exon_target = torch.tensor(
                [[lut.get(f"{m['event_id']}|{m['neuron']}", float("nan"))] for m in metadata],
                dtype=torch.float32, device=exon_pred.device)

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
        if psi_head == "twohead":
            perds.update(exon=exon_pred, dev=dev_pred, exon_target=exon_target)

        return perds, embed_list

    def loss(self, predictions, targets):

        psi_targets = self.transform_targets(
            targets,
        )

        psi_preds = predictions['psi']

        if getattr(self, "psi_head", "log") == "twohead":
            # exon (event-mean) + deviation losses on raw PSI, each scaled by its training variance;
            # rows missing from the lookup fall back to exon target = PSI (deviation 0)
            m = predictions['exon_target'].float()
            m = torch.where(torch.isnan(m), psi_targets, m)
            exon_loss = F.mse_loss(predictions['exon'].float(), m) / self.var_exon
            dev_loss = F.mse_loss(predictions['dev'].float(), psi_targets - m) / self.var_dev
            return exon_loss + dev_loss

        # main psi regression loss
        if self.psi_lossfn == "mse":
            psi_loss = nn.MSELoss(reduction="mean")(psi_preds, psi_targets)
        elif self.psi_lossfn == "l1":
            psi_loss = nn.L1Loss(reduction="mean")(psi_preds, psi_targets)

        return psi_loss

    def transform_targets(self, targets):
        psi_targets = targets['psi'].to(self.device)
        if getattr(self, "psi_head", "log") != "log":
            return psi_targets
        psi_targets = torch.log(psi_targets + self.adj_factor) - self.adj_factor
        return psi_targets

    def untransform_targets(self, predictions):
        if getattr(self, "psi_head", "log") != "log":
            return predictions['psi']
        psi_preds = torch.exp(predictions['psi'] + self.adj_factor) - self.adj_factor
        return psi_preds
 