"""Data, run folders and configs for the C. elegans ablation experiments (worm_train.py and friends).

Ablation inputs in the data root (<data_root>/c_elegans/processed):
  expression/ablations/<denoiser>/mean_vecs/                    MLP encoder input
  expression/ablations/<denoiser>/<metric>/scatter_coeffs*/     scatter encoder input (242 splicing factors)
  expression/ablations/<denoiser>/<metric>/graphs/              GNN encoder input
      denoiser: magic | raw_log1p | knn_smooth;  metric: mi_knn3 | mi_knn5 | mi_knn10 | mi_binned10 | pearson | spearman
  expression/scatter_243/                                       the published 243-factor per-neuron expression
  structure/structure_scattering_dict.pkl                       500-nt structure window (default)
  structure/len_<300|750|1000>/structure_scattering_dict.pkl    other window lengths
Runs: $CSN_ABL_RUNS (default <out>/worm_ablations/runs)/<group>/<name>__seed<N>/ with model/, ablation_config.json
and data_config.ini. Runs trained before the repo can be used by pointing CSN_ABL_RUNS at their folder; their
data paths are mapped onto the data root (legacy_path).
"""
import json
import os
import re
from pathlib import Path

from experiments.common.paths import WORM, derive_config, out_dir, split_config

EXPR_ABL = WORM / "processed/expression/ablations"
STRUCT = WORM / "processed/structure"
SCATTER_243 = WORM / "processed/expression/scatter_243"
DEFAULT_SPLIT = "rows"          # the original row split with control exons (the paper / rebuttal ablation data)


def abl_runs() -> Path:
    return Path(os.environ.get("CSN_ABL_RUNS") or out_dir("worm_ablations", "runs"))


def structure_pkl(length: int) -> Path:
    return STRUCT / "structure_scattering_dict.pkl" if int(length) == 500 else STRUCT / f"len_{length}/structure_scattering_dict.pkl"


_LEGACY = [
    (r".*/(?:processed/expression_ablations|preprocessing/ablations)/(magic|raw_log1p|knn_smooth)(/.*)?$",
     lambda m: str(EXPR_ABL / m.group(1)) + (m.group(2) or "")),
    (r".*/structure_ablations/241204/(\d+)/structure_scattering_dict(?:_\d+)?\.pkl$", lambda m: str(structure_pkl(m.group(1)))),
    (r".*/expression_dataset/pp04_scatter/?$", lambda m: str(SCATTER_243)),
]


def legacy_path(p):
    """Map a data path recorded by a pre-repo run onto the data root (unchanged if not recognised)."""
    p = str(p)
    for pat, f in _LEGACY:
        m = re.match(pat, p)
        if m:
            return f(m)
    return p


def split_ini(split):
    """A split name (data root, old names accepted) or a data_config.ini path."""
    return Path(split) if str(split).endswith(".ini") else split_config(split, "c_elegans")


def run_config(run_dir, split=None):
    """(ablation_config dict, data_config.ini) for a run. Runs made by worm_train.py carry their
    data_config.ini; for older runs a config is derived from `split` (default: $SPLIT, else 'rows') and the
    run's recorded expression / structure inputs, written to <out>/worm_ablations/configs/."""
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "ablation_config.json"))
    ini = run_dir / "data_config.ini"
    if ini.exists() and split is None:
        return cfg, ini
    split = split or os.environ.get("SPLIT") or DEFAULT_SPLIT
    files = {"structure_data_root": legacy_path(cfg["structure_pkl"]),
             "scatter_coeffs_dir": legacy_path(cfg["scatter_coeffs_dir"]),
             "mean_vec_dir": legacy_path(cfg["mean_vec_dir"]),
             "graph_metric_dir": legacy_path(cfg["graph_metric_dir"])}
    name = f"{run_dir.parent.name}__{run_dir.name}__{Path(str(split)).stem if str(split).endswith('.ini') else split}.ini"
    return cfg, derive_config(split_ini(split), out_dir("worm_ablations", "configs") / name, **files)


def run_argv(prog, cfg, ini, batch_size=None):
    """sys.argv for src/args.py argparser_fn, matching what worm_train.py used for the run."""
    return [prog, "--config_fname", str(ini), "--structure_length", str(cfg["structure_len"]),
            "--expression_encoder", str(cfg["expression_encoder"]), "--sfgenes", str(cfg.get("sfgenes", 242)),
            "--batch_size", str(batch_size or cfg.get("batch_size", 8)), "--lr", str(cfg.get("lr", 2e-5)),
            "--device", "cuda", "--eval_every_iteration", "500"]


# ----------------------------------------------------------------------------- rebuttal analyses
# old grouped / SE split names -> data-root splits (the 12Jan2026 *_single splits = *_sample1)
GROUPED = {"SE": "rows_se", "eventbased": "event_sample1", "genebased": "gene_sample1", "neuronbased": "cell_sample1"}
EVENTS_COORDS = WORM / "processed/events_coordinates.tsv"
SC_EXPR = WORM / "raw/expression"            # CeNGEN single-cell counts (241119b_scCounts_neuronsBulk.*)
GENE_IDS = WORM / "raw/c_elegans.PRJNA13758.WS291.geneIDs.txt"


def read_preds(path):
    """Per-row predictions with columns event_id, neuron, target, pred (src/eval.py files are renamed)."""
    import pandas as pd
    d = pd.read_csv(path)
    return d.rename(columns={"psi_target": "target", "psi_pred": "pred", "true_psi": "target", "pred_psi": "pred"})


def run_preds(group, split, seed, ckpt="best_validation_model"):
    """Predictions of run <group>/<split>__seed<seed>: dump_preds.py output (new or old split name), else the
    clean_preds.csv an older run dir carries. Returns the path (raises if none exists)."""
    old = {v: k for k, v in GROUPED.items()}
    names = [GROUPED.get(split, split), old.get(GROUPED.get(split, split), split)]
    pr = Path(os.environ.get("CSN_PREDS") or out_dir("worm_ablations", "preds"))
    cands = [pr / f"{group}__{n}__seed{seed}" / f"preds_{ckpt}.csv" for n in names]
    cands += [abl_runs() / group / f"{n}__seed{seed}" / "clean_preds.csv" for n in names]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"no predictions for {group}/{split} seed {seed}: run dump_preds.py ({cands[0]})")
