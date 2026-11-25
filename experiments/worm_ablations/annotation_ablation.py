"""Annotation-ablation test for CellSpliceNet (the untested confound, reviewer R2).

The exon/intron annotation marker is NOT a standalone modality: it is one-hot
concatenated into the PER-POSITION feature block of three modalities' embedding
inputs, so the existing whole-token perturbation harness never touches it alone.
This script zeros / shuffles ONLY the annotation feature columns (keeping the
nucleotide one-hot and the scattering coeffs intact), to ask how much of the
prediction is carried by the positional exon/intron marker rather than by the
sequence/structure content itself.

Per-position feature layout (annotation = trailing annot_vocab_size cols of each
per-position block; eff_window_size=1 for roi/structure so 1 block per token):
  sequence   : block = vocab_size + annot_vocab_size,  annot = last annot_vocab_size
  roi        : block = 6 + 4 = 10,                     annot = last 4
  structure  : token = 11 coeffs + 6 annot = 17,       annot = last 6

Conditions: clean, then knockout(zero) + shuffle(permute annot blocks across
positions, destroying exon/intron localization while preserving the marginal),
each applied to {sequence, roi, structure} individually and all-together.

Env: RUN_DIR, RESULTS_CSV(default <out>/worm_ablations/annotation/<group>__<run>.csv), SPLIT (older runs),
     TEST_MAX_SAMPLES(0=full), SEED(42), BATCH_SIZE, DEVICE, MIN_NEURONS(5).
"""
import sys, os, pathlib, json, csv, time, math
from collections import defaultdict
import numpy as np

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import run_argv, run_config

RUN_DIR = pathlib.Path(os.environ["RUN_DIR"]).resolve()
CKPT = RUN_DIR / "model" / os.environ.get("CKPT_NAME", "best_validation_model.pth")
CFG_FILE = RUN_DIR / "ablation_config.json"
RESULTS_CSV = pathlib.Path(os.environ.get("RESULTS_CSV", str(out_dir("worm_ablations", "annotation") / f"{RUN_DIR.parent.name}__{RUN_DIR.name}.csv")))
TEST_MAX_SAMPLES = int(os.environ.get("TEST_MAX_SAMPLES", "0"))
SEED = int(os.environ.get("SEED", "42"))
MIN_NEURONS = int(os.environ.get("MIN_NEURONS", "5"))

if not CKPT.exists():
    raise SystemExit(f"[annot] no checkpoint: {CKPT}")
if not CFG_FILE.exists():  # runs made by worm_train.py
    raise SystemExit(f"[annot] no ablation_config.json: {CFG_FILE}")

cfg, DATA_CONFIG = run_config(RUN_DIR)   # older runs: data config derived from $SPLIT (default rows)
config_name = RUN_DIR.name
BATCH_SIZE = os.environ.get("BATCH_SIZE", str(cfg.get("batch_size", 8)))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.argv = run_argv("annotation_ablation.py", cfg, DATA_CONFIG, BATCH_SIZE)

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from args import argparser_fn
from data.splicedata_dataloader import splicedata_dataloader
from eval import repoint_data_paths

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(DEVICE)

args = argparser_fn(dataset_type=cfg.get("dataset_tag", "worm_ablation"), batch_size=int(BATCH_SIZE))
args.device = DEVICE
data = splicedata_dataloader(args)
data.setup()
test_loader = data.test_dataloader()
args = data.setup_hparams(args)

model = torch.load(str(CKPT), weights_only=False, map_location=device)
if isinstance(model, nn.DataParallel):
    model = model.module
repoint_data_paths(model, args.dataset_paths, device)   # data files come from the config, not the pickle
model.to(device)
for _m in model.modules():
    if hasattr(_m, "device"):
        _m.device = device
model.eval()

enc = model.psi_transformer_encoder
seq_m, roi_m, str_m = enc.sequence_model, enc.local_roi_model, enc.local_structure_model

# (block_size, n_seq_feats) per modality: annotation = trailing (block_size - n_seq)
# columns of each per-position block; block repeats window times across the token.
LAYOUT = {
    "sequence":  (seq_m.vocab_size + seq_m.annot_vocab_size, seq_m.vocab_size),
    "roi":       (roi_m.vocab_size + roi_m.annot_vocab_size, roi_m.vocab_size),
    "structure": (11 + str_m.annot_vocab_size, 11),  # 11 scattering coeffs + 6 annot one-hot
}
EMB_MODULE = {
    "sequence":   seq_m.to_patch_embedding,
    "roi":        roi_m.to_local_embedding,
    "structure":  str_m.to_scattering_embedding,
}


def annot_cols(modality, D):
    """Indices within a token's flattened feature vector that hold annotation."""
    block, n_seq = LAYOUT[modality]
    if D % block != 0:
        raise RuntimeError(f"[annot] {modality}: dim {D} not divisible by block {block}")
    cols = [w * block + j for w in range(D // block) for j in range(n_seq, block)]
    return torch.tensor(cols, dtype=torch.long, device=device)


_COLCACHE = {}
STATE = {"armed": False, "mods": set(), "op": None, "gen": None}


def make_pre_hook(modality):
    def hook(module, inputs):
        if not STATE["armed"] or modality not in STATE["mods"]:
            return None
        x = inputs[0].clone()
        D = x.shape[-1]
        key = (modality, D)
        if key not in _COLCACHE:
            _COLCACHE[key] = annot_cols(modality, D)
        cols = _COLCACHE[key]
        if STATE["op"] == "knockout":
            x[:, :, cols] = 0
        else:  # shuffle annotation blocks across token positions, per sample
            g = STATE["gen"]
            L = x.shape[1]
            for b in range(x.shape[0]):
                perm = torch.randperm(L, generator=g, device=device)
                x[b][:, cols] = x[b][perm][:, cols]
        return (x,) + tuple(inputs[1:])
    return hook


for m in EMB_MODULE:
    EMB_MODULE[m].register_forward_pre_hook(make_pre_hook(m))


def to_dev(x):
    return [t.to(device) for t in x]


@torch.no_grad()
def forward(x, meta, mods, op, seed_off):
    STATE.update(armed=bool(mods), mods=set(mods), op=op,
                 gen=torch.Generator(device=device).manual_seed(SEED + seed_off))
    try:
        perds = model(to_dev(x), meta, embed_list_return=False)
        if isinstance(perds, tuple):
            perds = perds[0]
        preds = model.untransform_targets(perds).detach().cpu().squeeze(1)
    finally:
        STATE.update(armed=False, mods=set())
    return preds


CONDS = [("clean", [], None)]
for op in ["knockout", "shuffle"]:
    for mod in ["sequence", "roi", "structure"]:
        CONDS.append((f"{mod}", [mod], op))
    CONDS.append(("all", ["sequence", "roi", "structure"], op))

# de-dupe clean per op (clean is op-independent); keep a single clean entry
LABELS = [("clean", "-")] + [(mod, op) for (mod, mods, op) in CONDS if mods]

preds_acc = {(lab, op): [] for (lab, op) in LABELS}
targets, event_ids = [], []
t0 = time.time(); n_seen = 0
for it, (meta, x, y) in enumerate(test_loader):
    if TEST_MAX_SAMPLES and n_seen >= TEST_MAX_SAMPLES:
        break
    targets.append(y["psi"].detach().cpu().squeeze(1))
    event_ids.extend([m["event_id"] for m in meta])
    n_seen += y["psi"].shape[0]
    preds_acc[("clean", "-")].append(forward(x, meta, [], None, 0))
    for ci, (mod, mods, op) in enumerate(CONDS):
        if not mods:
            continue
        preds_acc[(mod, op)].append(forward(x, meta, mods, op, seed_off=1000 * ci + it))
    if it % 20 == 0:
        print(f"[annot] batch {it} n={n_seen} ({time.time()-t0:.0f}s)", flush=True)

t = torch.cat(targets).numpy()
ev_arr = np.array(event_ids)


def across_neuron_spearman(p):
    by = defaultdict(list)
    for i, e in enumerate(ev_arr):
        by[e].append(i)
    rs = []
    for e, idxs in by.items():
        if len(idxs) < MIN_NEURONS:
            continue
        pp = p[idxs]; tt = t[idxs]
        if np.unique(tt).size < 2 or np.unique(pp).size < 2:
            continue
        r = spearmanr(pp, tt).statistic
        if not math.isnan(r):
            rs.append(r)
    return (float(np.mean(rs)) if rs else float("nan")), len(rs)


clean_p = torch.cat(preds_acc[("clean", "-")]).numpy()
clean_sp = float(spearmanr(clean_p, t).statistic)
clean_an, n_ev = across_neuron_spearman(clean_p)
print(f"\n[annot] clean pooled={clean_sp:.4f} across-neuron={clean_an:.4f} "
      f"(events>= {MIN_NEURONS}n: {n_ev}) n={len(t)}", flush=True)

rows = []
for (lab, op) in LABELS:
    p = torch.cat(preds_acc[(lab, op)]).numpy()
    sp = float(spearmanr(p, t).statistic)
    pe = float(pearsonr(p, t).statistic)
    an, _ = across_neuron_spearman(p)
    rows.append(dict(checkpoint=config_name, seed=cfg.get("seed"),
                     target=lab, op=(op or "-"),
                     spearman=round(sp, 4), pearson=round(pe, 4),
                     across_neuron_spearman=round(an, 4),
                     drop_pooled=round(clean_sp - sp, 4),
                     drop_across_neuron=round(clean_an - an, 4), n=len(t)))

RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"\n[annot] === annotation ablation ({config_name}) ===")
print(f"{'target':<11}{'op':<10}{'pooled':>10}{'drop':>9}{'acrossN':>10}{'drop':>9}")
for r in rows:
    print(f"{r['target']:<11}{r['op']:<10}{r['spearman']:>10.4f}{r['drop_pooled']:>+9.4f}"
          f"{r['across_neuron_spearman']:>10.4f}{r['drop_across_neuron']:>+9.4f}")
print(f"\n[annot] wrote {RESULTS_CSV}", flush=True)
