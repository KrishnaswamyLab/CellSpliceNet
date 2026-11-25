"""Per-row test (or valid) predictions for ablation / rebuttal runs with src/eval.py.

    python experiments/worm_ablations/dump_preds.py <run dir> [<run dir> ...]
Env: SPLIT        split to predict on (default: the run's own data config; required for runs made before the
                  repo, e.g. rows_se, event_sample1 ...). Predicting on another split = cross-evaluation.
     CKPT_NAMES   checkpoint names in model/ (default best_validation_model; 'all' = best + every step_*)
     PRED_SPLIT   test | valid (default test);  OUT  output root (default <out>/worm_ablations/preds)
     MODEL, SEQONLY_SYNTH_SHAPES, SEQONLY_DROP_ANNOT, SEQONLY_DROP_ROI are passed through to the model code.
Writes <OUT>/<group>__<run>[__<split>]/preds_<ckpt>.csv (columns row, batch, gene_id, event_id, neuron,
psi_target, psi_pred; experiments.worm_ablations.common.read_preds renames them to target / pred).
"""
import glob, os, subprocess, sys
from pathlib import Path

from experiments.common.paths import REPO, out_dir
from experiments.worm_ablations.common import run_config

split = os.environ.get("SPLIT")
names = os.environ.get("CKPT_NAMES", "best_validation_model").split()
root = Path(os.environ.get("OUT") or out_dir("worm_ablations", "preds"))
for run in map(Path, sys.argv[1:]):
    if names == ["all"]:
        cks = sorted(glob.glob(str(run / "model/best_validation_model.pth")) + glob.glob(str(run / "model/step_*_model.pth")))
    else:
        cks = [str(run / f"model/{n}.pth") for n in names if (run / f"model/{n}.pth").exists()]
    if not cks:
        print(f"[skip] no checkpoint in {run}", flush=True); continue
    cfg, ini = run_config(run, split)
    out = root / (f"{run.parent.name}__{run.name}" + (f"__{split}" if split else ""))
    cmd = [sys.executable, str(REPO / "src/eval.py"), "--config_fname", str(ini), "--ckpts", *cks,
           "--split", os.environ.get("PRED_SPLIT", "test"), "--batch_size", str(cfg.get("batch_size", 8)),
           "--structure_length", str(cfg["structure_len"]), "--expression_encoder", cfg["expression_encoder"],
           "--sfgenes", str(cfg.get("sfgenes", 242)), "--out_dir", str(out)]
    print("[dump]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
