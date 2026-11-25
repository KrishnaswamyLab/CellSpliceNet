"""Attention-faithfulness INPUT-perturbation test for CellSpliceNet (reviewer R2).

Perturb the RAW INPUT at selected positions of each modality, re-run the FULL
model (non-circular), and measure the prediction drop. Faithful attention =>
perturbing TOP-attended inputs hurts >> perturbing BOTTOM-attended / RANDOM.

Two evaluation axes (both reported per condition):
  * pooled Spearman          -> dominated by between-event (sequence) variance
  * across-neuron Spearman   -> within-event / across cell type; the fair axis
                                for EXPRESSION (worm PSI is ~96% between-event,
                                ~4% within-event), averaged over events with
                                >= MIN_NEURONS neurons.

Selection modes:
  top / bottom / random      : by per-position attention, over valid positions
                               (all modalities + 'all')
  loop / stem                : STRUCTURE only -- perturb ALL unpaired '.' (loop)
                               vs ALL paired '()' (stem) positions (dot-bracket)
  loop_attn                  : STRUCTURE only -- top-attended positions WITHIN loops

Perturbation ops (on the raw featurized input row at selected positions):
  noise     : replace with Gaussian matched to that sample's per-feature stats
  shuffle   : permute the selected input rows among themselves
  knockout  : zero the input rows (remove the signal)

Hooks fire on each modality's input-embedding module (raw featurized input):
  sequence  to_patch_embedding | roi to_local_embedding |
  structure to_scattering_embedding | expression to_expression_embedding
Token position i of a modality's attention maps 1:1 to row i of its module input.

Env: RUN_DIR, RESULTS_CSV(default <out>/worm_ablations/perturb/<group>__<run>.csv), SPLIT (older runs),
     K_FRAC(0.2), TEST_MAX_SAMPLES(0=full), SEED(42), BATCH_SIZE, DEVICE,
     MIN_NEURONS(5)
"""
import sys, os, pathlib, json, csv, time, math
from collections import defaultdict
import numpy as np

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import run_argv, run_config

RUN_DIR = pathlib.Path(os.environ["RUN_DIR"]).resolve()
CKPT = RUN_DIR / "model" / os.environ.get("CKPT_NAME", "best_validation_model.pth")
CFG_FILE = RUN_DIR / "ablation_config.json"
RESULTS_CSV = pathlib.Path(os.environ.get("RESULTS_CSV", str(out_dir("worm_ablations", "perturb") / f"{RUN_DIR.parent.name}__{RUN_DIR.name}.csv")))
K_FRAC = float(os.environ.get("K_FRAC", "0.2"))
# Dose-response: partial doses swept for the attention-ranked modes (top/random).
# The 1.0 endpoint is added separately as a "full" mode (= 100% of valid positions,
# a test-time whole-modality knockout; no retraining). Override with K_FRACS env.
K_FRACS = [float(x) for x in os.environ.get("K_FRACS", "0.2,0.5").split(",")]
TEST_MAX_SAMPLES = int(os.environ.get("TEST_MAX_SAMPLES", "0"))
SEED = int(os.environ.get("SEED", "42"))
MIN_NEURONS = int(os.environ.get("MIN_NEURONS", "5"))

if not CKPT.exists():
    raise SystemExit(f"[perturb] no checkpoint: {CKPT}")
if not CFG_FILE.exists():  # runs made by worm_train.py
    raise SystemExit(f"[perturb] no ablation_config.json: {CFG_FILE}")

cfg, DATA_CONFIG = run_config(RUN_DIR)   # older runs: data config derived from $SPLIT (default rows)
config_name = RUN_DIR.name
BATCH_SIZE = os.environ.get("BATCH_SIZE", str(cfg.get("batch_size", 8)))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.argv = run_argv("perturb_faithfulness.py", cfg, DATA_CONFIG, BATCH_SIZE)

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from args import argparser_fn
from data.splicedata_dataloader import splicedata_dataloader
from eval import repoint_data_paths
from nn.roi import local_seq_extraction  # dict-metadata variant used by prep_local_sequence

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
MODALITIES = ["sequence", "structure", "roi", "expression"]
OPS = ["noise", "shuffle", "knockout"]

STATE = {"armed": False, "mods": set(), "op": None, "sel": {}, "gen": None}
EMB_MODULE = {
    "sequence":   enc.sequence_model.to_patch_embedding,
    "roi":        enc.local_roi_model.to_local_embedding,
    "structure":  enc.local_structure_model.to_scattering_embedding,
    "expression": enc.expression_model.to_expression_embedding,
}


def make_pre_hook(modality):
    def hook(module, inputs):
        if not STATE["armed"] or modality not in STATE["mods"]:
            return None
        x = inputs[0].clone()
        sel, op, g = STATE["sel"][modality], STATE["op"], STATE["gen"]
        for b in range(x.shape[0]):
            idx = sel[b]
            if idx.numel() == 0:
                continue
            if op == "knockout":
                x[b, idx] = 0
            elif op == "noise":
                mu = x[b].mean(0, keepdim=True)
                sd = x[b].std(0, keepdim=True).clamp_min(1e-6)
                x[b, idx] = mu + sd * torch.randn(idx.numel(), x.shape[-1],
                                                  generator=g, device=x.device, dtype=x.dtype)
            else:  # shuffle
                perm = idx[torch.randperm(idx.numel(), generator=g, device=x.device)]
                x[b, idx] = x[b, perm]
        return (x,) + tuple(inputs[1:])
    return hook


for m in MODALITIES:
    EMB_MODULE[m].register_forward_pre_hook(make_pre_hook(m))


def to_dev(x):
    return [t.to(device) for t in x]


@torch.no_grad()
def clean_forward(x, meta):
    perds, embed = model(to_dev(x), meta, embed_list_return=True)
    preds = model.untransform_targets(perds).detach().cpu().squeeze(1)
    attn = {
        "sequence":   embed["rna_attn"].float(),
        "roi":        embed["roi_attn"].float(),
        "structure":  embed["structurte_attn"].float(),
        "expression": embed["experssion_attn_post_transformer"].float(),
    }
    seq, annot = to_dev(x)
    seq_valid = enc.sequence_model.annotation_prep(seq, annot)[1].cpu()
    local_seq, _ = local_seq_extraction(seq, annot, meta, local_seq_length=enc.local_seq_length)
    roi_valid = (local_seq != enc.local_roi_model.pad_indx).cpu()
    struct_valid = (embed["structure_annotation"] != 0).cpu()
    exp_valid = torch.ones_like(attn["expression"], dtype=torch.bool)
    valid = {"sequence": seq_valid, "roi": roi_valid, "structure": struct_valid, "expression": exp_valid}
    for k in valid:  # clip to attn length if off-by-padding
        L = attn[k].shape[1]
        if valid[k].shape[1] != L:
            v = torch.zeros(attn[k].shape, dtype=torch.bool)
            n = min(L, valid[k].shape[1]); v[:, :n] = valid[k][:, :n]; valid[k] = v
    # loop/stem masks from the dot-bracket secondary structure (aligned to structure positions)
    L = attn["structure"].shape[1]
    ss = embed.get("secondary_structure", None)
    loop = torch.zeros(attn["structure"].shape, dtype=torch.bool)
    stem = torch.zeros(attn["structure"].shape, dtype=torch.bool)
    if ss is not None:
        for b, s in enumerate(ss):
            s = str(s)
            for i in range(min(L, len(s))):
                if s[i] == ".":
                    loop[b, i] = True
                elif s[i] in "()":
                    stem[b, i] = True
        loop &= valid["structure"]; stem &= valid["structure"]
    ev = [m["event_id"] for m in meta]
    return preds, attn, valid, loop, stem, ev


def pick(attn_b, cand_pos, k_frac, ranking, g):
    if cand_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if ranking == "all":
        return cand_pos.to(device)
    k = max(1, min(int(round(k_frac * cand_pos.numel())), cand_pos.numel()))
    a = attn_b[cand_pos]
    if ranking == "top":
        sel = cand_pos[torch.topk(a, k).indices]
    elif ranking == "bottom":
        sel = cand_pos[torch.topk(a, k, largest=False).indices]
    else:
        sel = cand_pos[torch.randperm(cand_pos.numel(), generator=g)[:k]]
    return sel.to(device)


# mode -> (candidate restriction, ranking).  "full" = ALL valid positions (the
# 100% test-time knockout endpoint of the dose-response curve).
def mode_spec(mode):
    return {
        "top":       ("valid", "top"),
        "bottom":    ("valid", "bottom"),
        "random":    ("valid", "random"),
        "full":      ("valid", "all"),
        "loop":      ("loop",  "all"),
        "stem":      ("stem",  "all"),
        "loop_attn": ("loop",  "top"),
    }[mode]


def select_mod(modality, mode, k_frac, attn, valid, loop, stem, g):
    restrict, ranking = mode_spec(mode)
    base = {"valid": valid[modality], "loop": loop, "stem": stem}[restrict]
    out = []
    for b in range(base.shape[0]):
        cand = torch.nonzero(base[b], as_tuple=False).squeeze(-1)
        out.append(pick(attn[modality][b], cand, k_frac, ranking, g))
    return out


@torch.no_grad()
def perturbed_forward(x, meta, mods, op, sel_by_mod, seed_off):
    STATE.update(armed=True, mods=set(mods), op=op,
                 sel={m: sel_by_mod[m] for m in mods},
                 gen=torch.Generator(device=device).manual_seed(SEED + seed_off))
    try:
        perds, _ = model(to_dev(x), meta, embed_list_return=False)
        preds = model.untransform_targets(perds).detach().cpu().squeeze(1)
    finally:
        STATE.update(armed=False, mods=set())
    return preds


# build condition list: (modality, mode, op, k_frac)
#   dose-response: top/random ranked at each partial dose in K_FRACS,
#   bottom control at the smallest dose, "full" = 100% of valid positions (1.0).
#   structure additionally gets loop/stem (full category) + loop_attn.
CONDS = [("clean", None, None, 0.0)]
for op in OPS:
    for mod in MODALITIES + ["all"]:
        for kf in K_FRACS:
            CONDS.append((mod, "top", op, kf))
            CONDS.append((mod, "random", op, kf))
        CONDS.append((mod, "bottom", op, K_FRACS[0]))
        CONDS.append((mod, "full", op, 1.0))
        if mod == "structure":
            CONDS.append((mod, "loop", op, 1.0))
            CONDS.append((mod, "stem", op, 1.0))
            CONDS.append((mod, "loop_attn", op, K_FRACS[0]))

preds_acc = {c: [] for c in CONDS}
targets, event_ids = [], []
cpu_gen = torch.Generator()
t0 = time.time(); n_seen = 0
for it, (meta, x, y) in enumerate(test_loader):
    if TEST_MAX_SAMPLES and n_seen >= TEST_MAX_SAMPLES:
        break
    clean_preds, attn, valid, loop, stem, ev = clean_forward(x, meta)
    targets.append(y["psi"].detach().cpu().squeeze(1)); event_ids.extend(ev)
    n_seen += y["psi"].shape[0]
    preds_acc[("clean", None, None, 0.0)].append(clean_preds)

    # selections cached per (modality, mode, k_frac) -- op-independent
    need = set()
    for (mod, mode, op, kf) in CONDS:
        if mod == "clean":
            continue
        for m in (MODALITIES if mod == "all" else [mod]):
            need.add((m, mode, kf))
    sel_cache = {}
    for (m, mode, kf) in need:
        cpu_gen.manual_seed(SEED + 7919 * it + (abs(hash((m, mode, kf))) % 100000))
        sel_cache[(m, mode, kf)] = select_mod(m, mode, kf, attn, valid, loop, stem, cpu_gen)

    for ci, (mod, mode, op, kf) in enumerate(CONDS):
        if mod == "clean":
            continue
        mods = MODALITIES if mod == "all" else [mod]
        sel_by_mod = {m: sel_cache[(m, mode, kf)] for m in mods}
        preds_acc[(mod, mode, op, kf)].append(
            perturbed_forward(x, meta, mods, op, sel_by_mod, seed_off=1000 * ci + it))
    if it % 10 == 0:
        print(f"[perturb] batch {it} n={n_seen} ({time.time()-t0:.0f}s)", flush=True)

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


clean_p = torch.cat(preds_acc[("clean", None, None, 0.0)]).numpy()
clean_sp = float(spearmanr(clean_p, t).statistic)
clean_an, n_ev = across_neuron_spearman(clean_p)
print(f"[perturb] clean pooled Spearman={clean_sp:.4f}  across-neuron Spearman={clean_an:.4f} "
      f"(events>= {MIN_NEURONS} neurons: {n_ev})  n={len(t)}", flush=True)

rows = []
for (mod, mode, op, kf) in CONDS:
    p = torch.cat(preds_acc[(mod, mode, op, kf)]).numpy()
    sp = float(spearmanr(p, t).statistic)
    pe = float(pearsonr(p, t).statistic)
    an, _ = across_neuron_spearman(p)
    rows.append(dict(checkpoint=config_name, seed=cfg.get("seed"), k_frac=kf,
                     modality=mod, mode=(mode or "-"), op=(op or "-"),
                     spearman=round(sp, 4), pearson=round(pe, 4),
                     across_neuron_spearman=round(an, 4),
                     drop_pooled=round(clean_sp - sp, 4),
                     drop_across_neuron=round(clean_an - an, 4), n=len(t)))

RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)


# dose-response columns: ranked top/random sweep, full=100% endpoint, bottom control,
# and the structure-only biological modes.
COLS = ([("top", kf) for kf in K_FRACS] + [("full", 1.0)] +
        [("random", kf) for kf in K_FRACS] + [("bottom", K_FRACS[0])] +
        [("loop", 1.0), ("stem", 1.0), ("loop_attn", K_FRACS[0])])


def show(metric_key, drop_key, title):
    print(f"\n[perturb] === {title} ===  ({config_name}, doses={K_FRACS}+full)")
    for op in OPS:
        print(f"\n-- op={op}: {metric_key} (drop vs clean) --")
        hdr = [f"{m}@{kf:g}" for (m, kf) in COLS]
        print(f"{'modality':<11}" + "".join(f"{h:>16}" for h in hdr))
        for mod in MODALITIES + ["all"]:
            cells = []
            for (m, kf) in COLS:
                r = next((r for r in rows if r["modality"] == mod and r["mode"] == m
                          and r["op"] == op and abs(r["k_frac"] - kf) < 1e-9), None)
                cells.append(f"{r[metric_key]:.3f}({r[drop_key]:+.3f})" if r else f"{'-':>14}")
            print(f"{mod:<11}" + "".join(f"{c:>16}" for c in cells))


print(f"\nclean: pooled={clean_sp:.4f}  across-neuron={clean_an:.4f}")
show("spearman", "drop_pooled", "POOLED Spearman (sequence/structure axis)")
show("across_neuron_spearman", "drop_across_neuron", "ACROSS-NEURON Spearman (expression / cell-type axis)")
print(f"\n[perturb] wrote {RESULTS_CSV}", flush=True)
