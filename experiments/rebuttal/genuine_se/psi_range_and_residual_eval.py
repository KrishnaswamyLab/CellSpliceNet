"""Reviewer-response eval: genuine SE events only, intermediate-PSI filter,
SE-only-trained vs all-types-trained model, and within-event (mean-subtracted)
neuron-dependent variation.

Addresses 4 reviewer asks in one pass:
  1. genuine skipped-exon (SE) events only  -> both input files already SE-only
     (SE-only clean_preds_SE.csv) or filtered here (all-types clean_preds_seed1.csv)
  2. intermediate PSI (0.05 < PSI < 0.95)   -> --psi_lo/--psi_hi row filter on target
  3. trained w/o controls vs trained w/ controls -> compares both prediction files
  4. within-event neuron-dependent variation after subtracting each event's mean PSI
     -> per-event mean-centered residuals, pooled across events, correlated

Inputs (defaults = existing seed-1 artifacts, see memory csn_se_only_vs_controls.md):
  SE-only-trained preds  : <preds>/se_only__magic_mi_knn3_scatter__seed<S>/preds_best_validation_model.csv
  all-types-trained preds: <preds>/expression__magic_mi_knn3_scatter__seed<S>__rows/... (filtered to SE_ prefix)
  (<preds> = <out>/worm_ablations/preds, written by run_seed.sbatch / dump_preds.py)
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import read_preds

P = out_dir("worm_ablations", "preds")


def load(path, se_prefix_only=False):
    d = read_preds(path)
    if se_prefix_only:
        d = d[d.event_id.str.startswith("SE")].copy()
    return d


def restrict_common(a, b):
    ka = set(zip(a.event_id, a.neuron))
    kb = set(zip(b.event_id, b.neuron))
    common = ka & kb
    a = a[a.apply(lambda r: (r.event_id, r.neuron) in common, axis=1)].copy() if False else a
    # faster: build key col
    a = a.assign(_k=list(zip(a.event_id, a.neuron)))
    b = b.assign(_k=list(zip(b.event_id, b.neuron)))
    a = a[a._k.isin(common)].drop(columns="_k").reset_index(drop=True)
    b = b[b._k.isin(common)].drop(columns="_k").reset_index(drop=True)
    return a, b, len(common)


def psi_filter(d, lo, hi):
    return d[(d.target > lo) & (d.target < hi)].copy()


def pooled_metrics(d):
    if len(d) < 2 or d.target.std() == 0 or d.pred.std() == 0:
        return dict(n=len(d), spearman=np.nan, pearson=np.nan, mae=np.nan)
    return dict(
        n=len(d),
        spearman=spearmanr(d.pred, d.target).statistic,
        pearson=pearsonr(d.pred, d.target).statistic,
        mae=float(np.abs(d.pred - d.target).mean()),
    )


def within_event_residuals(d, min_neurons):
    """Subtract each event's own mean (target & pred) then pool residuals
    across all qualifying events. Also return the per-event-averaged rho for
    reference (equivalent to raw per-event rho since centering doesn't change
    within-group rank)."""
    rows = []
    per_event_rhos = []
    for ev, g in d.groupby("event_id"):
        if g.neuron.nunique() < min_neurons:
            continue
        tr = g.target - g.target.mean()
        pr = g.pred - g.pred.mean()
        rows.append(pd.DataFrame({"event_id": ev, "t_resid": tr, "p_resid": pr}))
        if g.target.std() > 0 and g.pred.std() > 0:
            per_event_rhos.append(spearmanr(g.pred, g.target).statistic)
    if not rows:
        return dict(n_events=0, n_rows=0, pooled_resid_spearman=np.nan,
                     pooled_resid_pearson=np.nan, mean_per_event_rho=np.nan)
    pooled = pd.concat(rows, ignore_index=True)
    out = dict(
        n_events=pooled.event_id.nunique(),
        n_rows=len(pooled),
        pooled_resid_spearman=spearmanr(pooled.p_resid, pooled.t_resid).statistic
        if pooled.t_resid.std() > 0 and pooled.p_resid.std() > 0 else np.nan,
        pooled_resid_pearson=pearsonr(pooled.p_resid, pooled.t_resid).statistic
        if pooled.t_resid.std() > 0 and pooled.p_resid.std() > 0 else np.nan,
        mean_per_event_rho=float(np.nanmean(per_event_rhos)) if per_event_rhos else np.nan,
        median_target_resid_range=float(
            pooled.groupby("event_id").t_resid.apply(lambda s: s.max() - s.min()).median()
        ),
        median_pred_resid_range=float(
            pooled.groupby("event_id").p_resid.apply(lambda s: s.max() - s.min()).median()
        ),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--se_pred", default=None)
    ap.add_argument("--all_pred", default=None)
    ap.add_argument("--psi_lo", type=float, default=0.05)
    ap.add_argument("--psi_hi", type=float, default=0.95)
    ap.add_argument("--min_neurons", type=int, default=5)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    args.se_pred = args.se_pred or P / f"se_only__magic_mi_knn3_scatter__seed{args.seed}/preds_best_validation_model.csv"
    args.all_pred = args.all_pred or P / f"expression__magic_mi_knn3_scatter__seed{args.seed}__rows/preds_best_validation_model.csv"
    args.out_csv = args.out_csv or out_dir("rebuttal", "genuine_se") / f"psi_range_and_residual_eval_seed{args.seed}.csv"

    se = load(args.se_pred)
    al = load(args.all_pred, se_prefix_only=True)
    se, al, n_common = restrict_common(se, al)
    print(f"# genuine SE (event,neuron) rows, common to both pred files: {n_common}\n")

    records = []
    for filt_name, filt in [
        ("all_PSI (unfiltered)", (-np.inf, np.inf)),
        (f"intermediate_PSI ({args.psi_lo}-{args.psi_hi})", (args.psi_lo, args.psi_hi)),
    ]:
        lo, hi = filt
        for model_name, d in [("all_types_trained", al), ("SE_only_trained", se)]:
            dd = psi_filter(d, lo, hi)
            m = pooled_metrics(dd)
            r = within_event_residuals(dd, args.min_neurons)
            rec = dict(psi_filter=filt_name, model=model_name, **m, **r)
            records.append(rec)
            print(f"=== {filt_name} | {model_name} ===")
            print(f"  rows={m['n']}  pooled Spearman={m['spearman']:.4f}  Pearson={m['pearson']:.4f}  MAE={m['mae']:.4f}")
            print(f"  within-event (mean-subtracted) residuals, >={args.min_neurons} neurons/event, n_events={r['n_events']}, n_rows={r['n_rows']}:")
            print(f"    pooled residual Spearman={r['pooled_resid_spearman']:.4f}  Pearson={r['pooled_resid_pearson']:.4f}")
            print(f"    mean of per-event rho={r['mean_per_event_rho']:.4f}")
            print(f"    median residual range: target={r['median_target_resid_range']:.4f}  pred={r['median_pred_resid_range']:.4f}\n")

    out = pd.DataFrame.from_records(records)
    import os
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
