#!/usr/bin/env python3
"""Build the reviewer-R2 ablation results table from the worm CSN sweep.

Re-reads every loss_dict.json under <runs>/{expression,structure}/ each
run, so it is safe to re-invoke as the last jobs finish. Emits a markdown table
(grouped by ablation axis, anchor flagged) and a flat CSV.

A variant whose SLURM job is still in `squeue` is flagged PARTIAL — its best
valid loss is the best-so-far, not final.

Usage: python experiments/worm_ablations/make_r2_table.py
"""
import json, glob, os, subprocess

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import abl_runs
OUTDIR = str(abl_runs())

# variant -> (axis, label, is_anchor). Anchor = MAGIC + MI-kNN3 + scattering + 500bp.
SPEC = {
    "structure/stc_500":                  ("structure window", "500 bp (anchor)", True),
    "structure/stc_750":                  ("structure window", "750 bp", False),
    "structure/stc_1000":                 ("structure window", "1000 bp", False),
    "expression/magic_mi_knn3_scatter":   ("anchor",           "MAGIC + MI-kNN3 + scatter", True),
    "expression/raw_log1p_mi_knn3_scatter":("denoiser",        "raw log1p (no MAGIC)", False),
    "expression/knn_smooth_mi_knn3_scatter":("denoiser",       "kNN smoothing", False),
    "expression/magic_mi_knn5_scatter":   ("graph metric",     "MI-kNN5", False),
    "expression/magic_mi_knn10_scatter":  ("graph metric",     "MI-kNN10", False),
    "expression/magic_mi_binned10_scatter":("graph metric",    "MI-binned10", False),
    "expression/magic_pearson_scatter":   ("graph metric",     "Pearson", False),
    "expression/magic_spearman_scatter":  ("graph metric",     "Spearman", False),
    "expression/magic_mi_knn3_mlp":       ("encoder",          "MLP (mean-vec)", False),
    "expression/magic_mi_knn3_gnn":       ("encoder",          "GNN (GCNConv)", False),
}

AXIS_ORDER = ["anchor", "denoiser", "graph metric", "encoder", "structure window"]


def running_variants():
    """Map: variant key -> True if its SLURM job is still queued/running."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return set()
    names = set(out.split())
    live = set()
    for var in SPEC:
        # job names: csn_exp_<expr-variant>, csn_stc_<n>
        leaf = var.split("/")[1]
        jn_exp = "csn_exp_" + leaf
        jn_stc = "csn_" + leaf  # stc_750 -> csn_stc_750
        if jn_exp in names or jn_stc in names:
            live.add(var)
    return live


def best(f):
    d = json.load(open(f))
    v, t, s = d["valid_loss"], d["train_loss"], d["step"]
    bi = min(range(len(v)), key=lambda i: v[i])
    return {
        "best_valid": v[bi], "best_step": s[bi],
        "train_at_best": t[bi] if bi < len(t) else float("nan"),
        "last_step": s[-1], "last_valid": v[-1],
        "n_evals": len(v),
    }


def collect():
    live = running_variants()
    rows = []
    for var, (axis, label, anchor) in SPEC.items():
        f = os.path.join(OUTDIR, var, "model", "loss_dict.json")
        if not os.path.exists(f):
            rows.append(dict(var=var, axis=axis, label=label, anchor=anchor,
                             missing=True, partial=var in live))
            continue
        r = best(f)
        r.update(var=var, axis=axis, label=label, anchor=anchor,
                 missing=False, partial=var in live)
        rows.append(r)
    return rows


def detect_duplicate_curves():
    """Group variants whose valid_loss curve is byte-identical to another's.
    Identical curves => the two runs consumed identical model inputs => the
    ablated knob did not change what the model saw. Returns a list of groups
    (each a list of variant keys, in SPEC order)."""
    import hashlib
    spec_order = list(SPEC)
    by_hash = {}
    for var in spec_order:
        f = os.path.join(OUTDIR, var, "model", "loss_dict.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        h = hashlib.md5(json.dumps(d["valid_loss"]).encode()).hexdigest()
        by_hash.setdefault(h, []).append(var)
    groups = [sorted(g, key=spec_order.index) for g in by_hash.values() if len(g) > 1]
    return groups


# Known/explained collapses: identical curves we understand and report honestly,
# rather than treating as a surprise bug. Keyed by frozenset of variant keys.
# pearson == spearman: see EXPLAINED_FOOTNOTE.
EXPLAINED = {
    frozenset({"expression/magic_pearson_scatter",
               "expression/magic_spearman_scatter"}): "dense correlation → identical graph",
}

EXPLAINED_FOOTNOTE = (
    "† **Pearson and Spearman are reported as a single row** because they produce "
    "byte-identical results under this pipeline, and that is a real property of the "
    "published (unweighted) method, not a run we can separate. The geometric-scattering "
    "encoder reads only graph *connectivity* (`data.x`, `data.edge_index`) and discards "
    "edge weights (`gcn_norm` defaults `edge_weight` to all-ones). Pearson and Spearman "
    "similarity matrices are both fully dense, so after thresholding they yield the *same* "
    "`edge_index`; with identical node features the scattering coefficients come out "
    "byte-identical. The MI metrics (kNN3/5/10, binned) do *not* collapse, because the "
    "kNN-MI estimator emits metric-specific zeros that change which edges survive. "
    "Takeaway for R2: the metric axis tests **which edges survive sparsification**, not "
    "how edges are weighted — phrase the claim as graph *topology/sparsification* choice."
)


def fmt_md(rows, dups):
    anchor_best = next((r["best_valid"] for r in rows
                        if r.get("anchor") and not r.get("missing")), None)
    # Build merge map: within a same-axis duplicate group, collapse to the first
    # variant (representative) and suppress the rest. Only merge groups we
    # understand (EXPLAINED); anything else stays separate and gets a loud flag.
    suppress = set()
    merged_label = {}
    footnote_needed = False
    for g in dups:
        if frozenset(g) not in EXPLAINED:
            continue
        if len({SPEC[v][0] for v in g}) != 1:
            continue  # spans axes — don't merge
        rep, rest = g[0], g[1:]
        labels = " / ".join(SPEC[v][1] for v in g)
        merged_label[rep] = labels + " †"
        suppress.update(rest)
        footnote_needed = True

    # Surface any *unexplained* duplicate as a hard warning at the top.
    unexplained = [g for g in dups if frozenset(g) not in EXPLAINED]

    lines = ["# Reviewer R2 — worm CSN ablation results", ""]
    if unexplained:
        lines += ["> ⚠️ **Unexplained identical curves — investigate before using:**"]
        for g in unexplained:
            lines.append("> - " + " ≡ ".join(g))
        lines.append("")
    lines += [
        f"Anchor (best valid loss): **{anchor_best:.4f}**" if anchor_best else "",
        "Lower valid loss = better. Δ = best_valid − anchor (positive = worse than anchor).",
        "PARTIAL = SLURM job still running; value is best-so-far, not final.",
        ""]
    for axis in AXIS_ORDER:
        group = [r for r in rows if r["axis"] == axis and r["var"] not in suppress]
        if not group:
            continue
        lines.append(f"## {axis}")
        lines.append("")
        lines.append("| variant | best valid | @step | train@best | overfit gap | Δ vs anchor | last step | status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in group:
            label = merged_label.get(r["var"], r["label"])
            if r.get("missing"):
                lines.append(f"| {label} | — | — | — | — | — | — | "
                             f"{'PARTIAL/no evals yet' if r['partial'] else 'MISSING'} |")
                continue
            gap = r["best_valid"] - r["train_at_best"]
            delta = (r["best_valid"] - anchor_best) if anchor_best else float("nan")
            status = "PARTIAL" if r["partial"] else "done"
            mark = " ⭐" if r["anchor"] else ""
            lines.append(
                f"| {label}{mark} | {r['best_valid']:.4f} | {r['best_step']} | "
                f"{r['train_at_best']:.4f} | {gap:+.4f} | {delta:+.4f} | "
                f"{r['last_step']} | {status} |")
        lines.append("")
    if footnote_needed:
        lines.append(EXPLAINED_FOOTNOTE)
        lines.append("")
    return "\n".join(lines)


def fmt_csv(rows):
    hdr = ["axis", "label", "variant", "anchor", "best_valid", "best_step",
           "train_at_best", "last_step", "last_valid", "n_evals", "status"]
    out = [",".join(hdr)]
    for r in rows:
        if r.get("missing"):
            out.append(",".join([r["axis"], r["label"], r["var"], str(r["anchor"]),
                                  "", "", "", "", "", "0",
                                  "partial" if r["partial"] else "missing"]))
            continue
        out.append(",".join([
            r["axis"], r["label"], r["var"], str(r["anchor"]),
            f"{r['best_valid']:.6f}", str(r["best_step"]),
            f"{r['train_at_best']:.6f}", str(r["last_step"]),
            f"{r['last_valid']:.6f}", str(r["n_evals"]),
            "partial" if r["partial"] else "done"]))
    return "\n".join(out)


def main():
    rows = collect()
    dups = detect_duplicate_curves()
    md = fmt_md(rows, dups)
    csv = fmt_csv(rows)
    md_path = str(out_dir("worm_ablations") / "r2_results_table.md")
    csv_path = str(out_dir("worm_ablations") / "r2_results_table.csv")
    open(md_path, "w").write(md + "\n")
    open(csv_path, "w").write(csv + "\n")
    print(md)
    print(f"\nwrote {md_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
