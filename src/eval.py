"""Per-row predictions for one or more checkpoints (worm or human).

    python src/eval.py --config_fname <data_config.ini> --ckpts run/model/best_validation_model.pth [...] \
        [--split test|valid] [--max_events N] [--amp bf16] [--batch_size 16] [--out_dir DIR] \
        [any args.py flag, e.g. --sfgenes 493 --structure_length 500]

Data paths come from the config ([files], see utils/paths.py); model hyper-parameters come from the
pickled checkpoint, so only data-side flags (sfgenes, structure_length, expression_encoder, ...) matter.

Writes <out_dir or the checkpoint's run dir>/preds_<ckpt name>[_<split>].csv with columns
    row, batch, gene_id, event_id, neuron, psi_target, psi_pred [, exon_pred, dev_pred]
`row` is the index in the split's CSV and `batch` the batch number, so two evaluations can be
compared on identical rows and identical batch composition. Existing output files are skipped.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from args import argparser_fn
from data.splicedata_dataloader import splicedata_dataloader, sp3_collate_fn
from utils.validation import model_perd


def parse_eval_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint .pth files (pickled models)")
    p.add_argument("--split", default="test", choices=["test", "valid"])
    p.add_argument("--max_events", type=int, default=0,
                   help="evaluate all rows of this many randomly chosen events (RandomState(0)); 0 = all rows")
    p.add_argument("--amp", default=os.environ.get("AMP", ""), help="'bf16' for bfloat16 autocast")
    p.add_argument("--out_dir", default="", help="default: the checkpoint's run dir (parent of model/)")
    p.add_argument("--num_workers", type=int, default=6)
    a, _ = p.parse_known_args()
    return a


def main():
    ea = parse_eval_args()
    args = argparser_fn(dataset_type="eval")          # reads --config_fname, --batch_size, data flags
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    data = splicedata_dataloader(args)
    data.setup()
    args = data.setup_hparams(args)
    ds = data.test_data if ea.split == "test" else data.valid_data
    split_csv = args.dataset_paths.test_data_file if ea.split == "test" else args.dataset_paths.valid_data_file

    idx = list(range(len(ds)))
    if ea.max_events:
        ev_col = pd.read_csv(split_csv, usecols=["event_id"]).event_id
        assert len(ev_col) == len(ds), f"split CSV has {len(ev_col)} rows but dataset has {len(ds)}"
        ev = ev_col.unique()
        keep = set(np.random.RandomState(0).choice(ev, min(ea.max_events, len(ev)), replace=False))
        idx = [i for i, e in enumerate(ev_col) if e in keep]
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=ea.num_workers, persistent_workers=ea.num_workers > 0,
                        pin_memory=True, collate_fn=sp3_collate_fn)
    print(f"[eval] config={args.config_fname} split={ea.split} rows={len(idx)} batch={args.batch_size}", flush=True)

    for ck in ea.ckpts:
        name = os.path.basename(ck).replace(".pth", "") + ("" if ea.split == "test" else "_valid")
        out_dir = ea.out_dir or os.path.dirname(os.path.dirname(os.path.abspath(ck)))
        out = os.path.join(out_dir, f"preds_{name}.csv")
        if not os.path.exists(ck) or os.path.exists(out):
            print(f"[eval] skip {ck} (missing checkpoint or {out} exists)", flush=True)
            continue
        model = torch.load(ck, weights_only=False, map_location=device)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.to(device).eval()
        t0 = time.time()
        P, T, M, X, B = [], [], [], [], []
        with torch.no_grad():
            for b, (meta, x, y) in enumerate(loader):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ea.amp == "bf16"):
                    preds, _, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
                P.append(model.untransform_targets(preds).float().cpu().squeeze(1))
                T.append(y["psi"].cpu().squeeze(1))
                M += [(m["gene_id"], m["event_id"], m["neuron"]) for m in meta]
                B += [b] * len(meta)
                if "dev" in preds:
                    X.append(torch.cat([preds["exon"], preds["dev"]], 1).float().cpu())
        p = torch.cat(P).numpy(); t = torch.cat(T).numpy()
        xd = torch.cat(X).numpy().tolist() if X else None
        os.makedirs(out_dir, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "batch", "gene_id", "event_id", "neuron", "psi_target", "psi_pred"]
                       + (["exon_pred", "dev_pred"] if xd else []))
            for i, (r, bb, m, tt, pp) in enumerate(zip(idx, B, M, t.tolist(), p.tolist())):
                w.writerow([r, bb, *m, tt, pp] + (xd[i] if xd else []))
        print(f"[eval] {ck} n={len(p)} rho={spearmanr(t, p).correlation:.4f} -> {out} ({time.time() - t0:.0f}s)",
              flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("[eval] done", flush=True)


if __name__ == "__main__":
    main()
