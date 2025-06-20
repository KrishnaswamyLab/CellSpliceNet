from enum import Enum 
import torch 
import math 
from einops import rearrange, repeat, pack 

from nn.expression import GraphExpressionModality as ExpressionModality
from nn.sequence import EntireSeqModality
from nn.roi import LocalSeqModality
from nn.structure import StructureModality
import torchvision.utils as vutils
from nn.transformer_modules import LayerNorm, FeedForward, OutputHook, Attention_w_dropout, Attention
from torch import nn
  
  
class ModalTypes(Enum):
    SEQUENCE = 0
    LOCAL_SEQUENCE = 1
    SCATTRING_COEFF = 2
    EXPRESSION = 3
    FUSION = 4
    GLOBAL = 5 
    
  
class ZorroTransformer(nn.Module):
    def __init__(self, num_layers, hidden_dim, dim_head, nhead, device):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dim_head = dim_head
        self.nhead = nhead
        self.ff_mult = 4
        self.device = device

        self.norm = LayerNorm(self.hidden_dim) 
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention_w_dropout(dim=self.hidden_dim, dim_head=self.dim_head, heads=self.nhead), 
                FeedForward(dim=self.hidden_dim, mult=self.ff_mult)
            ]) for _ in range(num_layers)
        ]) 
  
    def forward(self, rna_tokens, roi_tokens, structure_tokens, expression_tokens, fusion_tokens):
        seq_tokens, roi_tokens, structure_tokens, expression_tokens, fusion_tokens = map(
            lambda t: rearrange(t, "b ... d -> b (...) d"),
            (rna_tokens, roi_tokens, structure_tokens, expression_tokens, fusion_tokens),
        )
    
        tokens, ps = pack(                   
            (
                seq_tokens,                  # [2, 391, 128]
                roi_tokens,                  # [2, 200, 128]
                structure_tokens,            # [2, 500, 128]
                expression_tokens,           # [2, 243, 128]       
                fusion_tokens,               # [2, 42,  128]            
            ),
            "b * d",
        ) 

        token_types = torch.tensor(
            list(
                (
                    *((ModalTypes.SEQUENCE.value,) * seq_tokens.shape[1]),
                    *((ModalTypes.LOCAL_SEQUENCE.value,) * roi_tokens.shape[1]),
                    *((ModalTypes.SCATTRING_COEFF.value,) * structure_tokens.shape[1]),
                    *((ModalTypes.EXPRESSION.value,) * expression_tokens.shape[1]),
                    *((ModalTypes.FUSION.value,) * fusion_tokens.shape[1]),
                )
            ),
            device=rna_tokens.device,  
            dtype=torch.long,
        ) 

        # boolean indices for each modality
        seq_token_bool = torch.eq(token_types, ModalTypes.SEQUENCE.value)
        local_seq_token_bool = torch.eq(token_types, ModalTypes.LOCAL_SEQUENCE.value)
        scattering_coeff_tokens_bool= torch.eq(token_types, ModalTypes.SCATTRING_COEFF.value)
        exp_token_bool = torch.eq(token_types, ModalTypes.EXPRESSION.value)
        fusion_token_bool = torch.eq(token_types, ModalTypes.FUSION.value)

        
        token_types_attend_from = rearrange(token_types, "i -> i 1")
        token_types_attend_to = rearrange(token_types, "j -> 1 j") 

        # every modality, including fusion can attend to self
        zorro_mask = token_types_attend_from == token_types_attend_to   
        
        # fusion can attend to everything
        zorro_mask_fusion = zorro_mask | token_types_attend_from == ModalTypes.FUSION.value 
        zorro_mask = zorro_mask | zorro_mask_fusion  
        # vutils.save_image(zorro_mask.float(), 'zorro_mask1.jpg')  

        num_fusion_params = fusion_token_bool.sum() 
        zorro_mask[exp_token_bool,:zorro_mask.shape[1]-num_fusion_params.item()] = 1  
        # vutils.save_image(zorro_mask.float(), 'zorro_mask2.jpg') 
        
        attn_out_list = []  
        for attn, ff in self.layers:  
            out_i, attn_out_val = attn(tokens, attn_mask=zorro_mask, return_attn=True)
            tokens = out_i + tokens
            attn_out_list.append(attn_out_val) 
            tokens = ff(tokens) + tokens 

        tokens = self.norm(tokens)   
         
        
        # separate each modality
        seq_tokens = tokens[:, seq_token_bool]
        local_seq_tokens = tokens[:, local_seq_token_bool]
        scattering_coeff_tokens = tokens[:, scattering_coeff_tokens_bool]
        exp_tokens = tokens[:, exp_token_bool]
        fusion_tokens = tokens[:, fusion_token_bool]

        return [seq_tokens, local_seq_tokens, scattering_coeff_tokens, exp_tokens, fusion_tokens], attn_out_list
    
class BioZorro(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Initialization of model parameters and tokens
        self.initialize_params(hparams)  
        if 'neuron_replicate' in hparams.dataset_type:
            from nn.expression_singlereplicant import ExpressionModality as ExpressionModality
        else: 
            from nn.expression import GraphExpressionModality as ExpressionModality

        # Initialize modality models
        self.sequence_model = EntireSeqModality(self.window_size, self.new_seq_length, self.eff_window_size, self.vocab_size, self.annot_vocab_size, self.hidden_dim, self.pad_indx)
        self.local_roi_model = LocalSeqModality(hparams, self.hidden_dim)
        self.expression_model = ExpressionModality(self.exp_dim, self.coeff_dim, self.hidden_dim, self.gene_embed_bool, self.bin_exp, self.ntype_feature_bool, hparams.expression_data_root, self.save_output_hook)
        self.local_structure_model = StructureModality(self.seq_coeff_dim, self.hidden_dim, structure_data_root=hparams.structure_data_root, structure_length=hparams.structure_length)
        self.ztransformer = ZorroTransformer(num_layers=hparams.layers, hidden_dim=self.hidden_dim, dim_head=64, nhead=hparams.nhead, device=self.device).to(self.device)

    def initialize_params(self, hparams):
        # Step and index initializations
        self.global_step = 0
        self.warmup_indx = 0

        # Parameter settings from hyperparameters
        self.num_fusion_tokens = 42
        self.window_size = hparams.window_size
        self.hidden_dim = hparams.hidden_dim
        self.pad_indx = hparams.dataparams.pad_indx
        self.vocab_size = hparams.dataparams.vocab_size
        self.annot_vocab_size = 4

        # Effective window size calculation
        self.max_seq_len = hparams.dataparams.max_prime_seq_len
        self.new_seq_length = math.ceil(self.max_seq_len / self.window_size)
        self.eff_window_size = torch.chunk(torch.ones(self.max_seq_len), self.new_seq_length)[0].shape[0]

        # Expression and other features
        self.bin_exp = hparams.bin_exp 
        self.ntype_feature_bool = hparams.ntype_feature_bool
        self.save_output_hook = OutputHook()
        self.gene_embed_bool = hparams.gene_embed_bool
        self.coeff_dim = 2662 #hparams.dataparams.coeff_dim
        self.exp_dim = 243 #242 #hparams.dataparams.exp_dim
        self.seq_coeff_dim = hparams.seq_coeff_dim
        self.device = hparams.device
        self.local_seq_length = hparams.local_seq_length
        self.fusion_tokens = nn.Parameter(torch.randn(self.num_fusion_tokens, self.hidden_dim)) 
        

    def forward(self, metadata, sequence_tensor, seq_annotation_tensor, embed_list_return=True): 
 
        # pre-embedding of the modalities  
        # i-1) whole sequence tokenization
        rna_tokens, glob_attn_mask = self.sequence_model.prep_sequence(
            sequence_tensor, 
            seq_annotation_tensor
        )        

        # i-2) roi sequence tokenization
        roi_tokens, roi_sequence = self.local_roi_model.prep_local_sequence(
            metadata, 
            sequence_tensor, 
            seq_annotation_tensor, 
            self.local_seq_length
        )

        # i-3) structure tokenization
        structure_tokens, structure_annotation, structure_seq, secondary_structure, structure_seq_letters = self.local_structure_model.prep_scattering_coeff(
            sequence_tensor, 
            metadata
        )                                                

        # i-4) expression tokenization
        expression_tokens, expression_tensor = self.expression_model.prep_expression(
            metadata
        )      
        
        # i-5) fusion tokenization
        fusion_tokens = repeat(
            self.fusion_tokens, "n d -> b n d", b = sequence_tensor.shape[0]
        )                                             
        
        # ii) transformer embedding    
        z_all, attn_out_list = self.ztransformer(
            rna_tokens, 
            roi_tokens, 
            structure_tokens, 
            expression_tokens, 
            fusion_tokens
        )  

        [
            z_rna_tokens, 
            z_roi_tokens, 
            z_structure_tokens, 
            z_expression_tokens, 
            z_fusion_tokens
        
        ] = z_all

        # iii) post-transformer attention pooling   
        # iii-1) whole sequence pooling
        z_rna_tokens_pool, z_rna_tokens_attn = self.sequence_model.seq_glob_attn_op(
            z_rna_tokens, 
            glob_attn_mask, 
            output_glob_attn=True
        )    

        # iii-2) roi sequence pooling 
        z_roi_tokens_pool, z_roi_tokens_att = self.local_roi_model.local_seq_glob_attn_op(
            z_roi_tokens, 
            output_glob_attn=True
        )

        # iii-3) structure pooling 
        z_structure_tokens_pool, z_structure_tokens_att = self.local_structure_model.scattering_coeff_glob_attn_op(
            z_structure_tokens, 
            output_glob_attn=True
        )

        # iii-4) expression pooling  
        z_expression_tokens_pool, z_expression_tokens_att = self.expression_model.exp_glob_attn_op(
            z_expression_tokens, 
            output_glob_attn=True
        ) 
 
        # fusion token output calculation     
        if embed_list_return: 
            structure_info = {
                'structure_annotation': structure_annotation.cpu().detach(),
                'structure_seq': structure_seq.cpu().detach(),
                'secondary_structure':secondary_structure, 
                'structure_seq_letters': structure_seq_letters,
            }  
          
            attn_out_data = { 
                "roi_sequence":roi_sequence.cpu(),
                "rna_attn": z_rna_tokens_attn.cpu(), 
                "roi_attn": z_roi_tokens_att.cpu(), 
                "structurte_attn": z_structure_tokens_att.cpu(),
                'experssion_attn_post_transformer': z_expression_tokens_att.cpu(), 
                'transformer_attention': attn_out_list[-1].mean(0).mean(0).cpu(), #[-1][:,:,attn_out_list[-1].shape[-2]-z_fusion_tokens.shape[1]:, :].sum(1).cpu(), 
                'fusion_transformer_attention': attn_out_list[-1][:,:,attn_out_list[-1].shape[-2]-z_fusion_tokens.shape[1]:, :].sum(1).cpu(),  
                'structure_info': structure_info
            }   
 
            output_embeds = [  
                z_rna_tokens,
                z_roi_tokens,
                z_expression_tokens,
                fusion_tokens,
            ]  
        else: 
            output_embeds = None
            attn_out_data = None 
         
        output_attention_pooling = [
            z_rna_tokens_pool, 
            z_roi_tokens_pool, 
            z_structure_tokens_pool, 
            z_expression_tokens_pool, 
            z_fusion_tokens.sum(1), 
        ]  
 
        return [
            output_attention_pooling, 
            expression_tensor, 
            attn_out_data, 
            output_embeds
        ]