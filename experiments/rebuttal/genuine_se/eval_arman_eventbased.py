"""Apply the reviewer-response eval (intermediate-PSI filter + mean-subtracted
within-event residual) to Arman's eventbased epoch_270 checkpoint, whose native
dataset (12Jan2026_eventbased_single = data-root split event_sample1) is 100% SE /
0% CE-CI by construction -- genuine-SE-only and no-controls-in-training are already
satisfied, no ablation needed. This is the checkpoint with real 243-factor per-neuron
splice-factor expression (see memory downloaded_arman_worm_models.md) that resolves
within-event/cell-type variation far better than our synthetic-expression retrain.

Predictions: run_seed.sbatch (ARMAN=1) writes them with src/eval.py from
<data_root>/c_elegans/checkpoints/arman_12Jan2026/event_sample1_epoch270.pth; PRED overrides the file.
"""
import os
from experiments.common.paths import out_dir
from experiments.rebuttal.genuine_se.psi_range_and_residual_eval import pooled_metrics, within_event_residuals, psi_filter
from experiments.worm_ablations.common import read_preds

PRED = os.environ.get("PRED") or str(out_dir("rebuttal", "arman", "event_sample1") / "preds_event_sample1_epoch270.csv")


def main():
    d = read_preds(PRED)
    assert d.event_id.str.startswith("SE").all(), "expected 100% SE, dataset lineage changed?"
    print(f"n={len(d)} (100% genuine SE, no CE/CI in this dataset lineage)\n")

    records = []
    for filt_name, (lo, hi) in [("all_PSI", (-1e9, 1e9)), ("intermediate_PSI_0.05_0.95", (0.05, 0.95))]:
        dd = psi_filter(d, lo, hi)
        m = pooled_metrics(dd)
        r = within_event_residuals(dd, min_neurons=5)
        records.append(dict(psi_filter=filt_name, model="arman_eventbased_epoch270_real_expr", **m, **r))
        print(f"=== {filt_name} (n={m['n']}) ===")
        print(f"  pooled Spearman={m['spearman']:.4f} Pearson={m['pearson']:.4f} MAE={m['mae']:.4f}")
        print(f"  within-event residual (>=5 neurons, n_events={r['n_events']}): "
              f"pooled resid Spearman={r['pooled_resid_spearman']:.4f} "
              f"Pearson={r['pooled_resid_pearson']:.4f} mean-per-event-rho={r['mean_per_event_rho']:.4f}")
        print()

    out = pd.DataFrame.from_records(records)
    out_csv = str(out_dir("rebuttal", "genuine_se") / "arman_eventbased_epoch270_eval.csv")
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
