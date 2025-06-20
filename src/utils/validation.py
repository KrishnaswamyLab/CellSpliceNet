import torch 
import numpy as np 
from viz.training_scatter import scatter_plot
from model import model_registry
from data.splicedata_dataloader import splicedata_dataloader
from args import argparser_fn, print_gpu_info, make_directroy
from viz.training_gene_along_neuron_plots import gene_along_neuron_plots
from torch.optim import lr_scheduler 
import shutil
import time
import os
import pickle
from data.splicedata_dataloader import local_seq_extraction

torch.manual_seed(1337)
np.random.seed(42)

def extention(args):
    if args.no_sequence_for_ablation:
        extention = 'no_seq'
    if args.no_structure_for_ablation:
        extention = 'no_stc'
    if args.no_roi_for_ablation:
        extention = 'no_roi'
    if args.no_expression_for_ablation:
        extention = 'no_exp' 
    else:
        extention = '' 
    return extention

def feedforwad_or_load(file_path, dataloder, model,  visualization=False, attention_out=False):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            embed_attention_dictionary = pickle.load(f)
    else: 
        loss_val_current, embed_attention_dictionary = validation_for_visualization(model, dataloder, visualization, attention_out) 
        with open(file_path, 'wb') as f:
            pickle.dump(embed_attention_dictionary, f)  

    return embed_attention_dictionary
    
def model_load(args, model_key='240910--20241104-210254', model_label='current_model', print_text='QUNTIFICATION  FIGUIRE CREATOR', dataset_type=None, fig=False):
  
    model_key = args.model+'-'+model_key
    # create dataloader   
    data = splicedata_dataloader(args)
    data.setup()
    train_dataloader = data.train_dataloader()  
    valid_dataloader = data.valid_dataloader()  
    test_dataloader = data.test_dataloader()  

    print_gpu_info(model_key+('\n             '+print_text)) 
    name_extention = extention(args)
    if args.fusion_residual: 
        name_extention += '_fusion_residual'
    
    if fig:
        model_key_model = '../../outputs/'+args.model+'/'+model_key+name_extention+'/'  
    else:
        model_key = '../../outputs/'+args.model+'/'+model_key+name_extention+'/'  
  
    model = torch.load(model_key_model+'model/'+model_label+'.pth', weights_only=False)    
    model.to(args.device)
    return model, model_key, train_dataloader, valid_dataloader, test_dataloader
 
## prediction 
def model_perd(model, data, metadata, targets, device, embed_list_return):  
    data = [x.to(device) for x in data]  
    model.to(device) 
    preds, embed_list = model(data, metadata, embed_list_return=embed_list_return)  
    loss = model.loss(predictions=preds, targets=targets)  
    return preds, loss, embed_list
  
## validation 
def validation_for_model(model, dataloder): 
    loss = [] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()   
 
    psi_preds_list, psi_targets_list = [], [] 
    delta_psi_preds_list, delta_psi_targets_list = [], [] 
 
    with torch.no_grad(): 
        for it, (metadata, data, targets) in enumerate(dataloder):  
            
            for metadata_i in metadata:  
                event_id = metadata_i["event_id"]
                gene_id = metadata_i["gene_id"]
                neuron = metadata_i["neuron"]
 
                preds, loss_i, embed_list_i = model_perd(model, data, metadata, targets, device, embed_list_return=False) 
                psi_preds = model.untransform_targets(preds)   
                loss.append(loss_i.cpu().numpy()) 
    
            psi_preds_list.append(psi_preds.detach().cpu().squeeze(1))  
            psi_targets_list.append(targets['psi'].detach().cpu().squeeze(1))      
   
    psi_preds_list = torch.cat(psi_preds_list)
    psi_targets_list = torch.cat(psi_targets_list) 
    
    preds_list = psi_preds_list  
    targets_list = psi_targets_list
    
    return sum(np.array(loss))/len(loss), preds_list, targets_list  


## validation 
def validation_for_visualization(model, dataloder, visualization=False, attention_out=False, taget_type='psi'):  
    model.eval()   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    embed_attention_dictionary = {
        'seq_embedding': [],
        'local_seq_embedding': [], 
        'structure_embedding': [],
        'expression_embedding': [],
        'residual_embedding': [],
        'transformer_zrep': [],
        'comb_zrep': [], 
        'seq_attn': [],
        'local_seq_attn': [],
        'pre_exp_embedding': [],
        'post_exp_embedding': [],
        'expression_attn': [], 
        'gene_id_list': [],
        'event_name_list': [],
        'neuron_name_list': [], 
        'preds': [], 
        'targets': [],
        'loss':[],
        'exp_pre_att':[],
        'roi_letters':[], 
        'fusion_transformer_attention':[],
        'transformer_attention':[],
        'structure_attention':[],
        'structure_annotation':[],
        'structure_seq':[],
        'secondary_structure':[],
        'structure_seq_letters':[],
    }  

    for it, (metadata, data, targets) in enumerate(dataloder):   

        # for event_id, event_name, _, neuron_name, corr in metadata: 
        for metadata_i in metadata:  
            event_id = metadata_i["event_id"]
            gene_id = metadata_i["gene_id"]
            neuron_name = metadata_i["neuron"]
            embed_attention_dictionary['gene_id_list'].append(event_id)
            embed_attention_dictionary['event_name_list'].append(gene_id)
            embed_attention_dictionary['neuron_name_list'].append(neuron_name)  

        with torch.no_grad():  
            preds, loss_i, embed_list_i = model_perd(model, data, metadata, targets, device, embed_list_return=True)  
            psi_preds = model.untransform_targets(preds)  
            embed_attention_dictionary['loss'].append(loss_i.cpu().numpy())  

        embed_attention_dictionary['targets'].append(psi_preds.detach().cpu().squeeze(1)) 
        embed_attention_dictionary['preds'].append(targets['psi'].detach().cpu().squeeze(1))

        if visualization:  
            embed_attention_dictionary['seq_embedding'].append(embed_list_i['z_rna_tokens_pool'].cpu())
            embed_attention_dictionary['local_seq_embedding'].append(embed_list_i['z_roi_tokens_pool'].cpu())
            embed_attention_dictionary['structure_embedding'].append(embed_list_i['z_structure_tokens_pool'].cpu())
            embed_attention_dictionary['expression_embedding'].append(embed_list_i['z_expression_tokens_pool'].cpu())
            embed_attention_dictionary['residual_embedding'].append(embed_list_i['z_residual_tokens_pool'].cpu())
            embed_attention_dictionary['transformer_zrep'].append(embed_list_i['transformer_zrep'].cpu()) 
            embed_attention_dictionary['comb_zrep'].append(embed_list_i['comb_zrep'].cpu())  
                  
            embed_attention_dictionary['structure_attention'].append(embed_list_i['structurte_attn'].cpu()) 
            embed_attention_dictionary['fusion_transformer_attention'].append(embed_list_i['fusion_transformer_attention'].cpu())
            embed_attention_dictionary['transformer_attention'].append(embed_list_i['transformer_attention'].cpu())
            embed_attention_dictionary['expression_attn'].append(embed_list_i['experssion_attn_post_transformer'].cpu()) 
            embed_attention_dictionary['local_seq_attn'].append(embed_list_i['roi_attn'].cpu()) 
            embed_attention_dictionary['seq_attn'].append(embed_list_i['rna_attn'].cpu())   
             
            embed_attention_dictionary['structure_annotation'].append(embed_list_i['structure_annotation']) 
            embed_attention_dictionary['structure_seq'].append(embed_list_i['structure_seq'])   
            embed_attention_dictionary['secondary_structure'].append(embed_list_i['secondary_structure'])    
            embed_attention_dictionary['structure_seq_letters'].append(embed_list_i['structure_seq_letters']) 
 
            local_seq = embed_list_i['roi_sequence'] 
            # vocab_map = {"PAD": 0, "A": 1, "G": 2, "U": 3, "C": 4, "X": 5}
            divocab_map = {"P": 0, "A": 1, "G": 2, "U": 3, "C": 4, "X": 5}
            reverse_map = {v: k for k, v in divocab_map.items()}  
            # 
            roi_letters_batch = [] 
            for row in range(local_seq.shape[0]):
                local_seq_r = local_seq[row, :]
                letter_list = [] 
                for col in range(local_seq_r.shape[0]):
                    letter_list.append(reverse_map[int(local_seq_r[col].item())])   
                roi_letters_batch.append(letter_list)
 
            embed_attention_dictionary['roi_letters'].append(roi_letters_batch) 
 
    # embed_attention_dictionary['exp_pre_att'] = torch.cat(embed_attention_dictionary['exp_pre_att'], dim=0)
    embed_attention_dictionary['targets'] = torch.cat(embed_attention_dictionary['targets'], dim=0)
    embed_attention_dictionary['preds'] = torch.cat(embed_attention_dictionary['preds'], dim=0)
    embed_attention_dictionary['gene_id_set'] = list(set(embed_attention_dictionary['gene_id_list']))
    embed_attention_dictionary['event_name_set'] = list(set(embed_attention_dictionary['event_name_list']))
    embed_attention_dictionary['neuron_name_set']= list(set(embed_attention_dictionary['neuron_name_list']))   
    
    preds_nocontrols, targets_nocontrols = [], []
    neuron_nocontrol_name_list = []
    for i, event_i in enumerate(embed_attention_dictionary['event_name_list']): 
        if event_i[0:2] != 'CE':
            preds_nocontrols.append(embed_attention_dictionary['preds'][i].unsqueeze(0))  
            targets_nocontrols.append(embed_attention_dictionary['targets'][i].unsqueeze(0))  
            neuron_nocontrol_name_list.append(embed_attention_dictionary['neuron_name_list'][i])
 
    embed_attention_dictionary['preds_nocontrols'] = torch.cat(preds_nocontrols)
    embed_attention_dictionary['targets_nocontrols'] = torch.cat(targets_nocontrols) 
    embed_attention_dictionary['neuron_nocontrol_name_list'] = neuron_nocontrol_name_list
    # print('-----------')
    return sum(np.array(embed_attention_dictionary['loss']))/len(embed_attention_dictionary['loss']), embed_attention_dictionary