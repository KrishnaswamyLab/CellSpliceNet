import argparse
from argparse import ArgumentParser
import torch
import os



def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)
    
def meg2List(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(item + '\n')
    f.close()

def make_directroy(args):  
    root_dir = '../../outputs/'
    mkdir_fun(root_dir)

    root_dir = os.path.join(root_dir+args.model)
    mkdir_fun(root_dir)
    
    root_dir = root_dir+'/'+ args.output_key+args.dataset_type
    mkdir_fun(root_dir) 
     
    mkdir_fun(os.path.join(root_dir, 'model'))    
    mkdir_fun(os.path.join(root_dir, 'scatter_valid'))   
    mkdir_fun(os.path.join(root_dir, 'scatter_valid/psi')) 
    mkdir_fun(os.path.join(root_dir, 'scatter_valid/delta-psi')) 
    mkdir_fun(os.path.join(root_dir, 'codes'))  
    # mkdir_fun(os.path.join(root_dir, 'scatter_pergene'))   
    return root_dir 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def print_gpu_info(args):
    os.system('cls' if os.name == 'nt' else 'clear')
    if torch.cuda.is_available():
        print("----------------------------------------------------------------")
        print()
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device) 
            print(f'              Device {i}:  ')
            print(f'                  Name: {props.name} ')
            print(f'                  Memory: {props.total_memory / 1024 ** 3:.2f} GB') 
            print()
    else:
        print('No GPU available.')

    print(f'              model: {args.output_key}') 
    print(f'              data_type: {args.dataset_type}') 
    print(f'              batch_size: {args.batch_size}') 
    print()
    print("----------------------------------------------------------------") 
    
def argparser_fn(dataset_type, batch_size=40, server='misha'):  
    parser = ArgumentParser(add_help=True) 
    if server == 'mccleary':
        dataset_root = '/home/aa2793/project/genenn/dataset/'
    elif server == 'misha': 
        dataset_root = '/gpfs/radev/home/aa2793/project/cellnn/dataset/'

    # data argmuments
    parser.add_argument("--model", default="deltacellsplicenet", type=str) 
    # parser.add_argument("--dataset", default="240910", type=str)  
    parser.add_argument("--dataset_type", default='01Feb2025_'+dataset_type, type=str)  
 
    parser.add_argument("--dataset_root", default=dataset_root, type=str) 
    parser.add_argument("--config_fname", default=dataset_root+'Alternative-Splicing/01Feb2025_'+dataset_type+'/data_config.ini', type=str)
 
    structure_length = 500
    parser.add_argument("--structure_length", default=structure_length, type=int)
    parser.add_argument("--structure_data_root", default=dataset_root+'Alternative-Splicing/main/structure_coeffs/structure_scattering_dict_'+str(structure_length)+'.pkl', type=str) 
    parser.add_argument("--expression_data_root", default=dataset_root+'Alternative-Splicing/main/full_data_Feb01.tsv', type=str) 
    
    parser.add_argument("--input_dim", default=5, type=int)
    parser.add_argument("--latent_dim", default=256, type=int)      
    parser.add_argument("--hidden_dim", default=512, type=int)          # 512
    parser.add_argument("--embedding_dim", default=256, type=int)
    parser.add_argument("--layers", default=12, type=int)              #     12
    parser.add_argument("--nhead", default=8, type=int)                #    8
    parser.add_argument("--probs", default=0.2, type=float)

    # loss weighting
    parser.add_argument("--alpha", default=5.0, type=float)
    parser.add_argument("--beta", default=0.001, type=float)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--delta", default=0.0, type=float)
    parser.add_argument("--ent_weight", default=0.0, type=float)

    # logging
    parser.add_argument("--tag_list", default=None, nargs="+")

    # data
    parser.add_argument("--no_ci", default=False, type=str2bool)
    parser.add_argument("--data_size", default=None, type=int)

    # sequence
    parser.add_argument("--max_prime_seq_len", default=25000, type=int)    # 30000
    parser.add_argument("--max_sj_seq_len", default=500, type=int)
    parser.add_argument("--window_size", default=64, type=int)  
    parser.add_argument("--seq_scale", default=1.0, type=float)

    seq_coeff_dim = 500
    parser.add_argument("--seq_coeff_dim", default=500, type=int) 
    parser.add_argument("--local_seq_length", default=200, type=int)
    
    # expression
    parser.add_argument("--bin_exp", default=False, type=str2bool)
    parser.add_argument("--exp_scale", default=1.0, type=float)
    parser.add_argument("--exp_norm_bool", default=False, type=str2bool)

    parser.add_argument("--topkgenes_bool", default=False, type=str2bool) ## top highly variable genes
    parser.add_argument("--topkgenes", default=5, type=int)
    parser.add_argument("--sfgenes_bool", default=True, type=str2bool)
    parser.add_argument("--sfgenes", default=243, type=int)

    # parser.add_argument("--total_genes", default=-1, type=int)

    # auxiliary arguments
    parser.add_argument("--gene_embed_bool", default=True, type=str2bool)
    parser.add_argument("--ntype_feature_bool", default=True, type=str2bool)
    parser.add_argument("--aux_neuron_loss", default="none", type=str)  
    
    # targets
    # parser.add_argument("--norm_psi", default=False, type=str2bool)
    parser.add_argument("--transform_type", default="log", type=str)
    parser.add_argument("--scale_targets", default=False, type=str2bool)
    parser.add_argument("--scale_value", default=1.0, type=float)
    parser.add_argument("--psi_lossfn", default="mse", type=str)
    parser.add_argument("--lr_schedule", default="none", type=str)
    # parser.add_argument("--delta_noise", default=False, type=str2bool)
    parser.add_argument("--shift_targets", default=True, type=str2bool)
    parser.add_argument("--nb_norm", default=False, type=str2bool)

    # training arguments
    parser.add_argument("--batch_size", default=batch_size, type=int)              # 40
    parser.add_argument("--eval_every_iteration", default=500, type=int)              # 16
    
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--project_name", default="splicenn", type=str)

    parser.add_argument("--resume", default=False, type=str2bool)
    parser.add_argument("--resume_weights", default=None, type=str)

    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--n_epochs", default=400, type=int)
    parser.add_argument("--n_steps", default=200000, type=int)
    parser.add_argument("--n_gpus", default=1, type=int)
    parser.add_argument("--dev", default=False, type=str2bool)

    parser.add_argument("--residual_expresion", default=False, type=str2bool)

    
    # parser.add_argument("--cpu", default=False, type=str2bool)

    parser.add_argument("--accum_grad_batch", default=64, type=int)
    parser.add_argument("--grad_clip_val", default=1, type=float)
    parser.add_argument("--final_eval", default=True, type=str2bool)

    parser.add_argument("--total_steps", default=None, type=int)
    parser.add_argument("--device", default='cuda')
      
    # ablation arguments
    parser.add_argument("--no_sequence_for_ablation", default=False, type=str2bool)
    parser.add_argument("--no_structure_for_ablation", default=False, type=str2bool)
    parser.add_argument("--no_roi_for_ablation", default=False, type=str2bool)
    parser.add_argument("--no_expression_for_ablation", default=False, type=str2bool)
    parser.add_argument("--fusion_residual", default=False, type=str2bool)
    # validation model key    
    # key = '20250111-223731' 
    # parser.add_argument("--fusion_residual", default=True, type=str2bool)
 
    # key = '20250127-131226'
    key = '20250128-162732'
    parser.add_argument("--model_key", default='240910--'+key) 
    args_out = parser.parse_args()
    # args_out.model = 'cellsplicenet'   
    args_out.model = 'deltacellsplicenet'  
    return  args_out
  
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