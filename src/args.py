import argparse
from argparse import ArgumentParser
import torch
import os

from utils.paths import preset_config_for_tag, resolve_training_paths, run_output_dir, ensure_run_output_layout


def meg2List(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(item + '\n')
    f.close()

def make_directroy(args):
    output_dir = run_output_dir(args.model, args.output_key, args.dataset_type)
    ensure_run_output_layout(output_dir)
    return str(output_dir)

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

def argparser_fn(dataset_type, batch_size=40):
    parser = ArgumentParser(add_help=True)
    preset_config = preset_config_for_tag(dataset_type)

    # data argmuments
    parser.add_argument("--model", default="deltacellsplicenet", type=str)
    # parser.add_argument("--dataset", default="240910", type=str)
    parser.add_argument("--dataset_type", default='01Feb2025_'+dataset_type, type=str)

    parser.add_argument("--config_fname", default=str(preset_config), type=str)

    structure_length = 500
    parser.add_argument("--structure_length", default=structure_length, type=int)

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
    parser.add_argument(
        "--sfgenes",
        default=243,
        type=int,
        help="Splice-factor genes in scatter .pt. Default 243 = worm (paper baseline); for GTEx pass --sfgenes 493 (must match preprocessing/expression_gtex build_meta.json n_genes).",
    )
    parser.add_argument(
        "--expression_encoder",
        default="scatter",
        type=str,
        choices=["scatter", "mlp", "gnn"],
        help="Which expression encoder to use: scatter (paper anchor, geometric scattering), mlp (per-gene mean vector), gnn (GCN over MI/corr graph).",
    )

    # Step-based trainer (train_full.py) run-budget / logging knobs. Each
    # defaults to an env var so sbatch wrappers that export them keep working;
    # pass the flag to override.
    parser.add_argument(
        "--time_budget_s",
        type=float,
        default=float(os.environ.get("TIME_BUDGET_S", str(5 * 3600 + 50 * 60))),
        help="Wallclock budget in seconds; train_full.py stops cleanly before the SLURM walltime (default ~5h50m).",
    )
    parser.add_argument(
        "--valid_max_batches",
        type=int,
        default=int(os.environ.get("VALID_MAX_BATCHES", "200")),
        help="Cap validation iterations per eval so each validation pass stays short (train_full.py).",
    )
    parser.add_argument(
        "--save_step_ckpt_every",
        type=int,
        default=int(os.environ.get("SAVE_STEP_CKPT_EVERY", "10")),
        help="Save a step checkpoint every Nth validation (best+final always saved); 0 disables (train_full.py).",
    )
    parser.add_argument(
        "--log_every_iters",
        type=int,
        default=int(os.environ.get("LOG_EVERY_ITERS", "50")),
        help="Per-iteration progress log cadence in steps (train_full.py).",
    )
    parser.add_argument(
        "--log_every_secs",
        type=float,
        default=float(os.environ.get("LOG_EVERY_SECS", "30")),
        help="Per-iteration progress log cadence in seconds (train_full.py).",
    )

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
    args_out = parser.parse_known_args()[0]
    # args_out.model = 'cellsplicenet'
    args_out.model = 'deltacellsplicenet'
    return resolve_training_paths(args_out)

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
