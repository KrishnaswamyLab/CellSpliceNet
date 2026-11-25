"""Aggregate SeqOnly runs against the full model, with the uncertainty conventions
used in the reviewer response.

Reports, per grouped split:
  * mean +/- SD across seeds (training variability), when >1 seed is present
  * a per-event Fisher-z 95% CI (one independent unit per test event) -- validated
    against an event-clustered bootstrap, which it matches at 0.76-0.97x
  * the event-clustered bootstrap CI itself, when per-row predictions exist

Usage: python experiments/rebuttal/aggregate_seqonly.py [--group seqonly|fullmodel_grouped]
           [--splits event_sample1 gene_sample1 cell_sample1] [--seeds 1 2 3] [--out CSV]
Predictions: experiments/worm_ablations/dump_preds.py output for <runs>/<group>/<split>__seed<N>.
"""
import argparse, os, glob
import numpy as np, pandas as pd
from scipy.stats import rankdata

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import read_preds, run_preds

RNG = np.random.default_rng(20260916)


def spearman(x, y):
    if len(x) < 3:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (rx.std() * ry.std()))


def fisher_ci(rho, n_events, z=1.96):
    """Per-event Fisher-z interval: each test event counts as one independent unit."""
    if not np.isfinite(rho) or n_events < 5:
        return (np.nan, np.nan)
    se = 1.0 / np.sqrt(n_events - 3)
    a = np.arctanh(rho)
    return float(np.tanh(a - z * se)), float(np.tanh(a + z * se))


def bootstrap_ci(d, B=2000):
    o = np.argsort(d.event_id.values, kind="stable")
    ev = d.event_id.values[o]
    b = np.flatnonzero(np.r_[True, ev[1:] != ev[:-1], True])
    bl = [o[b[i]:b[i + 1]] for i in range(len(b) - 1)]
    if len(bl) < 5:
        return (np.nan, np.nan)
    x, y = d.pred.values, d.target.values
    v = np.empty(B)
    for i in range(B):
        idx = np.concatenate([bl[j] for j in RNG.integers(0, len(bl), len(bl))])
        v[i] = spearman(x[idx], y[idx])
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["event_sample1", "gene_sample1", "cell_sample1"])
    ap.add_argument("--group", default="seqonly", help="run group under <runs>, e.g. fullmodel_grouped")
    ap.add_argument("--label", default="SeqOnly")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for split in a.splits:
        for seed in a.seeds:
            try:
                p = run_preds(a.group, split, seed)
            except FileNotFoundError:
                continue
            d = read_preds(p)
            rho = spearman(d.pred.values, d.target.values)
            nev = d.event_id.nunique()
            rows.append(dict(split=split, seed=seed, n=len(d), n_events=nev, rho=rho,
                             mae=float(np.abs(d.pred - d.target).mean())))
    if not rows:
        print(f"No predictions for group {a.group}. Train with worm_ablations/launch.sh, "
              "then run worm_ablations/dump_preds.py."); return
    df = pd.DataFrame(rows)

    print(f"[{a.label}] group={a.group}")
    print(f"{'split':14s} {'seeds':>6s} {'rho mean':>9s} {'SD':>7s} "
          f"{'n':>6s} {'events':>7s} {'Fisher 95% CI':>20s} {'bootstrap 95% CI':>22s}")
    for split, g in df.groupby("split"):
        r = g.rho.values
        nev = int(g.n_events.iloc[0])
        flo, fhi = fisher_ci(r.mean(), nev)
        blo, bhi = bootstrap_ci(read_preds(run_preds(a.group, split, int(g.seed.iloc[0]))))
        sd = r.std(ddof=1) if len(r) > 1 else np.nan
        print(f"{split:14s} {len(r):6d} {r.mean():9.4f} "
              f"{(f'{sd:.4f}' if len(r)>1 else 'n/a'):>7s} "
              f"{int(g.n.iloc[0]):6d} {nev:7d} "
              f"{f'[{flo:.3f}, {fhi:.3f}]':>20s} {f'[{blo:.3f}, {bhi:.3f}]':>22s}")
    out = a.out or str(out_dir("rebuttal") / f"{a.group}_summary.csv")
    a.out = out
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    print("\nFull-model reference (Arman's reported values, per-event Fisher CI):")
    for axis, rho, nev in [("neuron-held-out", 0.86, 440), ("gene-held-out", 0.86, 123),
                           ("event-held-out", 0.88, 133)]:
        lo, hi = fisher_ci(rho, nev)
        print(f"  {axis:16s} rho={rho:.2f}  95% CI [{lo:.3f}, {hi:.3f}]  ({nev} events)")


if __name__ == "__main__":
    main()
