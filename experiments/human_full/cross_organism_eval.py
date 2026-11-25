"""Cross-organism transfer eval for CellSpliceNet.

Loads a trained full-model checkpoint (worm- or human/GTEx-trained) and
evaluates it on EITHER organism's test split, to fill a 2x2 transfer matrix
(model-organism x data-organism).

The only organism-sized layer is the expression/cell-type tower's input
projection (493*11 GTEx vs 242*11 worm scatter coeffs), and the cell-type
identity enters the model ONLY through that tower. So a zero-shot cross-organism
eval must neutralise that tower. We do it SYMMETRICALLY in all four cells so the
diagonal and off-diagonal are directly comparable:

  MODE=seqstruct : zero the expression tower only  (sequence + structure transfer)
  MODE=seqonly   : zero expression AND structure    (sequence-only transfer)

Everything else (sequence tower, structure tower, fusion attention, 1-dim PSI
head -- 193/194 weight tensors) is shape-identical across organisms and
transfers verbatim. For the structure tower we point structure_data_root at the
TARGET organism's pickle; load_structure's built-in coord-consistency check
guarantees the metadata and pickle are paired correctly.

Env vars:
  CKPT              path to a full-model .pth (required)
  DATASET           gtex | worm   (which test split to evaluate on)
  MODE              seqstruct (default) | seqonly
  TEST_MAX_SAMPLES  default 20000; 0 = full test split
  BATCH_SIZE        default 8
  SEED              default 42
  TAG               label for the output dir (default derived)
"""
import sys, os, pathlib, time, glob, csv
import numpy as np

from experiments.common.paths import out_dir

CKPT     = os.path.abspath(os.environ["CKPT"])
DATASET  = os.environ.get("DATASET", "gtex").lower()
MODE     = os.environ.get("MODE", "seqstruct").lower()
BATCH    = os.environ.get("BATCH_SIZE", "8")
SEED     = int(os.environ.get("SEED", "42"))
TEST_MAX = int(os.environ.get("TEST_MAX_SAMPLES", "20000"))
TAG      = os.environ.get("TAG", f"{DATASET}_{MODE}")

assert DATASET in ("gtex", "worm")
assert MODE in ("seqstruct", "seqonly")
if not os.path.exists(CKPT):
    raise SystemExit(f"CKPT not found: {CKPT}")

# ---- per-dataset wiring ----------------------------------------------------
# human: the gene split (493 splicing factors); worm: the row split with controls and the 242-factor
# magic / mi_knn3 scatter input of the ablation runs. Structure: each organism's 500 nt dictionary.
from experiments.common.paths import derive_config, split_config
from experiments.worm_ablations.common import EXPR_ABL
if DATASET == "gtex":
    config_fname        = str(split_config(os.environ.get("SPLIT", "gene")))
    sfgenes             = "493"
    dataset_tag         = "gtex"
else:
    config_fname        = str(derive_config(split_config(os.environ.get("SPLIT", "rows"), "c_elegans"),
                                            out_dir("cross_organism", "configs") / "worm_242.ini",
                                            scatter_coeffs_dir=EXPR_ABL / "magic/mi_knn3/scatter_coeffs"))
    sfgenes             = "242"
    dataset_tag         = "worm_ablation"
structure_len = "500"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.argv = [
    "cross_organism_eval.py",
    "--config_fname",         config_fname,
    "--structure_length",     structure_len,
    "--sfgenes",              sfgenes,
    "--batch_size",           BATCH,
    "--device",               "cuda",
]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from args import argparser_fn
from eval import repoint_data_paths
from data.splicedata_dataloader import splicedata_dataloader, sp3_collate_fn
from utils.validation import model_perd

args = argparser_fn(dataset_type=dataset_tag, batch_size=int(BATCH))
args.device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)

print(f"[xorg] CKPT={CKPT}")
print(f"[xorg] DATASET={DATASET}  MODE={MODE}  device={device}")

data = splicedata_dataloader(args); data.setup()
args = data.setup_hparams(args)
n_test = len(data.test_data)
print(f"[xorg] test_N total = {n_test}")

if TEST_MAX > 0 and TEST_MAX < n_test:
    rng = np.random.RandomState(SEED)
    indices = rng.choice(n_test, size=TEST_MAX, replace=False).tolist()
    sampler = SubsetRandomSampler(indices)
    print(f"[xorg] sampling {TEST_MAX} of {n_test} rows (seed={SEED})")
else:
    sampler = SubsetRandomSampler(list(range(n_test)))
    print(f"[xorg] evaluating all {n_test} rows (shuffled)")

test_loader = DataLoader(
    data.test_data, batch_size=int(BATCH), sampler=sampler,
    num_workers=8, persistent_workers=True, pin_memory=True,
    collate_fn=sp3_collate_fn, drop_last=False,
)

# ---- load model + neutralise the non-transferable tower(s) -----------------
model = torch.load(CKPT, weights_only=False, map_location=device)
if isinstance(model, nn.DataParallel):
    model = model.module
repoint_data_paths(model, args.dataset_paths, device)   # data files come from the config
model.to(device); model.eval()

bz = model.psi_transformer_encoder if hasattr(model, "psi_transformer_encoder") else model
expr = bz.expression_model
struct = bz.local_structure_model

# Point structure lookups at the TARGET organism's pickle (overrides the path
# baked in at training time). Coord-consistency check in load_structure will
# fail loudly if metadata and pickle are mismatched.
struct.structure_data_root = str(args.dataset_paths.structure_data_root)

# Zero the expression/cell-type tower: feed a single zero token through the
# model's own to_expression_embedding (keeps weights/LayerNorm in the graph but
# the input -- hence the cell-type signal -- is zero). Bypasses scatter loading
# and the dim-mismatched Linear entirely.
_expr_dim = expr.expression_dim
_expr_lin = expr.to_expression_embedding
def _zero_prep_expression(metadata, _self=expr, _dim=_expr_dim, _lin=_expr_lin):
    dev = next(_lin.parameters()).device
    B = len(metadata)
    zin = torch.zeros(B, 1, _dim, device=dev)
    embd = _lin(zin)                       # [B, 1, hidden]
    ctx  = torch.zeros(B, 1, 1, device=dev)
    return embd, ctx
expr.prep_expression = _zero_prep_expression
print(f"[xorg] expression tower ZEROED (model expr_dim={_expr_dim})")

if MODE == "seqonly":
    # Run the real structure prep (so token count / annotations / masks stay
    # exactly as the model expects), then zero the structure embedding output.
    _orig_struct_prep = struct.prep_scattering_coeff
    def _zero_struct_prep(sequence_tensor, metadata, _orig=_orig_struct_prep):
        embed, ann, sseq, sec, letters = _orig(sequence_tensor, metadata)
        return torch.zeros_like(embed), ann, sseq, sec, letters
    struct.prep_scattering_coeff = _zero_struct_prep
    print("[xorg] structure tower ZEROED (sequence-only)")

# ---- eval loop -------------------------------------------------------------
t0 = time.time()
preds_all, targets_all, meta_all = [], [], []
n_batches = len(test_loader)
with torch.no_grad():
    for it, (meta, x, y) in enumerate(test_loader):
        preds, loss_i, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
        psi_preds = model.untransform_targets(preds).detach().cpu().squeeze(1)
        psi_targets = y["psi"].detach().cpu().squeeze(1)
        preds_all.append(psi_preds)
        targets_all.append(psi_targets)
        for m in meta:
            meta_all.append({k: m.get(k) for k in ("gene_id", "event_id", "neuron")})
        if it % 50 == 0:
            print(f"[xorg] batch {it}/{n_batches}  elapsed={time.time()-t0:.0f}s", flush=True)

preds_t  = torch.cat(preds_all).numpy()
target_t = torch.cat(targets_all).numpy()
n = len(preds_t)
print(f"[xorg] done. {n} predictions in {time.time()-t0:.0f}s")

mse = float(((preds_t - target_t) ** 2).mean())
mae = float(np.abs(preds_t - target_t).mean())
try:
    from scipy.stats import pearsonr, spearmanr
    pearson  = float(pearsonr(preds_t, target_t).statistic)
    spearman = float(spearmanr(preds_t, target_t).statistic)
except Exception:
    pearson = float(np.corrcoef(preds_t, target_t)[0, 1]); spearman = float("nan")

print(f"\n[xorg] === {TAG} (n={n}) ===")
print(f"  MSE      : {mse:.4f}")
print(f"  MAE      : {mae:.4f}")
print(f"  Pearson  : {pearson:.4f}")
print(f"  Spearman : {spearman:.4f}")

odir = out_dir("cross_organism", TAG)
with open(odir / "test_preds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gene_id", "event_id", "neuron", "psi_target", "psi_pred"])
    for m, t_, p_ in zip(meta_all, target_t.tolist(), preds_t.tolist()):
        w.writerow([m["gene_id"], m["event_id"], m["neuron"], t_, p_])
import json
with open(odir / "metrics.json", "w") as f:
    json.dump({"tag": TAG, "ckpt": CKPT, "dataset": DATASET, "mode": MODE,
               "n": n, "mse": mse, "mae": mae, "pearson": pearson,
               "spearman": spearman, "test_max": TEST_MAX, "seed": SEED}, f, indent=2)
print(f"[xorg] wrote {odir}/metrics.json")
