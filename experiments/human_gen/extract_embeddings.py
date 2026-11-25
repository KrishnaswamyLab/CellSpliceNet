"""Cache per-event embeddings from a stage-A (expression-free) exon model.

For every event in the gene_x_cell_em_all test file (train + valid + test exons, one row each) runs the
checkpoint and stores the input of the PSI head (z_head, the representation the exon/base prediction is
read from) plus the base prediction.  csn_noexpr zeroes the expression modality, so the embedding is
tissue-independent and one forward pass per event is enough.
Env: RUNS (glob rest after 'hg_gene_x_cell_em_', default csn_noexpr_em_s), CKPT (default best_validation_model.pth),
     AMP (bf16 for bfloat16 autocast; the stage-A runs were trained with it)
Writes <run>/event_embeddings_<ckpt>.npz with event_id, emb (float16), base_pred, psi_target.
    PSI_HEAD=linear AMP=bf16 sbatch experiments/common/gpu.sbatch experiments/human_gen/extract_embeddings.py
"""
import os, sys, time
import numpy as np

from experiments.common.paths import split_config, split_runs

BATCH = 16
sys.argv = ["extract_embeddings.py", "--config_fname", str(split_config("gene_x_cell_em_all")),
            "--sfgenes", "493", "--batch_size", str(BATCH), "--device", "cuda"]
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from args import argparser_fn
from data.splicedata_dataloader import splicedata_dataloader, sp3_collate_fn
from eval import repoint_data_paths
from utils.validation import model_perd
from scipy.stats import spearmanr

args = argparser_fn(dataset_type="eval", batch_size=BATCH)
device = torch.device("cuda")
data = splicedata_dataloader(args); data.setup(); args = data.setup_hparams(args)
loader = DataLoader(data.test_data, batch_size=BATCH, shuffle=False, num_workers=6, pin_memory=True,
                    collate_fn=sp3_collate_fn)
runs = split_runs("hg", "gene_x_cell_em", os.environ.get("RUNS", "csn_noexpr_em_s"))
ck = os.environ.get("CKPT", "best_validation_model.pth")
print(f"[emb] runs={runs} ckpt={ck} events={len(data.test_data)}", flush=True)
for run in runs:
    out = f"{run}/event_embeddings_{ck.replace('.pth', '')}.npz"
    if os.path.exists(out): continue
    model = torch.load(f"{run}/model/{ck}", weights_only=False, map_location=device)
    if isinstance(model, nn.DataParallel): model = model.module
    repoint_data_paths(model, args.dataset_paths, device)
    model.to(device).eval()
    cap = []
    h = model.psi_pred_nn.register_forward_hook(lambda m, i, o: cap.append(i[0].detach().float().cpu()))
    E, B, T, ids = [], [], [], []; t0 = time.time()
    with torch.no_grad():
        for k, (meta, x, y) in enumerate(loader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=os.environ.get("AMP", "") == "bf16"):
                preds, _, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
            E.append(cap.pop()); B.append(model.untransform_targets(preds).float().cpu().squeeze(1))
            T.append(y["psi"].cpu().squeeze(1)); ids += [m["event_id"] for m in meta]
            if k % 500 == 0: print(f"  {k * BATCH} events {time.time() - t0:.0f}s", flush=True)
    h.remove()
    E = torch.cat(E).numpy(); B = torch.cat(B).numpy(); T = torch.cat(T).numpy()
    np.savez_compressed(out, event_id=np.array(ids), emb=E.astype(np.float16), base_pred=B, psi_target=T)
    print(f"[emb] {os.path.basename(run)} emb={E.shape} rho(base, mean PSI)={spearmanr(B, T).correlation:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del model; torch.cuda.empty_cache()
