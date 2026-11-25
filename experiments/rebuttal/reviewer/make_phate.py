"""PHATE recoloring of the CellSpliceNet latent manifold (reviewer response).

For each model latent dump (worm eventbased / worm neuronbased / human GTEx),
compute a 2D PHATE embedding of the fused latent (transformer_zrep) and color it by:
  (a) ground-truth PSI   (b) predicted PSI   (c) absolute error
  (d) cell type (neuron / tissue)            (e) structural feature: intron length
  (f) read depth (worm)  /  exon length (human, no read-depth retained)

Also prints a QUANTIFICATION block answering "what does the manifold encode":
  - |Spearman(PHATE1, X)| for X in {PSI, error, intron len, read depth}
  - silhouette of cell-type labels on PHATE coords + kNN cell-type accuracy
    (near chance => manifold does NOT organize by cell type)

Run under `mioflow` (has phate). Outputs -> outputs/reviewer_phate/.
"""
import os, pathlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import phate
from experiments.common.paths import HUMAN, gtex_counts_dir, out_dir, split_csv

OUT = out_dir("reviewer")

DATASETS = {
    "worm_eventbased": {
        "npz": OUT / "worm_eventbased_emb.npz",
        "csv": split_csv("event_sample1", "test", "c_elegans"),
        "title": "Worm CellSpliceNet — event-disjoint split",
        "ctype": "neuron type", "has_reads": True,
    },
    "worm_neuronbased": {
        "npz": OUT / "worm_neuronbased_emb.npz",
        "csv": split_csv("cell_sample1", "test", "c_elegans"),
        "title": "Worm CellSpliceNet — neuron-disjoint split (held-out cell types)",
        "ctype": "neuron type", "has_reads": True,
    },
    "human_gtex": {
        "npz": OUT / "human_gtex_emb.npz",
        "csv": split_csv("gene", "test"),
        "title": "Human CellSpliceNet — GTEx",
        "ctype": "tissue", "has_reads": False,
    },
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def merge_features(d, csv):
    df = pd.read_csv(csv)
    df["exon_len"] = (df["exon_end"] - df["exon_start"]).abs()
    df["intron_len"] = (df["intron_end"] - df["intron_start"]).abs()
    key = lambda e, n: f"{e}||{n}"
    lut = {key(e, n): (er, il, el, gl) for e, n, er, il, el, gl in
           zip(df.event_id, df.neuron, df.nb_reads, df.intron_len, df.exon_len, df.gene_length)}
    reads, intron, exon, glen = [], [], [], []
    for e, n in zip(d["event_id"], d["neuron"]):
        v = lut.get(key(e, n), (np.nan, np.nan, np.nan, np.nan))
        reads.append(v[0]); intron.append(v[1]); exon.append(v[2]); glen.append(v[3])
    return (np.array(reads, float), np.array(intron, float),
            np.array(exon, float), np.array(glen, float))


def scatter(ax, Y, c, title, cmap, cbar_label=None, categorical=False, legend=False, norm=None):
    if categorical:
        cats = sorted(set(c))
        cmapd = plt.get_cmap("tab20", max(len(cats), 3))
        cidx = {k: i for i, k in enumerate(cats)}
        col = [cmapd(cidx[x]) for x in c]
        ax.scatter(Y[:, 0], Y[:, 1], c=col, s=6, linewidths=0, alpha=0.85)
        if legend:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", ls="", ms=5, mfc=cmapd(cidx[k]), mec="none")
                       for k in cats]
            ax.legend(handles, cats, frameon=False, fontsize=5, ncol=2,
                      loc="upper left", bbox_to_anchor=(1.0, 1.0))
    else:
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=c, s=6, linewidths=0, alpha=0.85, cmap=cmap, norm=norm)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PHATE 1", fontsize=7); ax.set_ylabel("PHATE 2", fontsize=7)


def drop_outliers(Y, thresh=6.0):
    """Robust (MAD) mask of points close to the bulk of the 2D embedding, so a
    stray PHATE outlier doesn't dominate the view. Returns a boolean keep-mask."""
    med = np.median(Y, axis=0)
    d = np.linalg.norm(Y - med, axis=1)
    mad = np.median(np.abs(d - np.median(d))) or 1.0
    z = 0.6745 * (d - np.median(d)) / mad
    return z < thresh


summary = {}
for name, cfg in DATASETS.items():
    print(f"\n=== {name} ===", flush=True)
    zrep = np.load(OUT / f"{name}_zrep.npy")
    meta = pd.read_csv(OUT / f"{name}_meta.csv")
    tgt, pred = meta["target"].to_numpy(), meta["pred"].to_numpy()
    neuron = meta["neuron"].astype(str).to_numpy()
    d = {"event_id": meta["event_id"].astype(str).to_numpy(), "neuron": neuron}
    err = np.abs(pred - tgt)
    reads, intron, exon, glen = merge_features(d, cfg["csv"])

    Z = StandardScaler().fit_transform(zrep)
    op = phate.PHATE(n_components=2, n_jobs=-1, verbose=0, random_state=0)
    Y = op.fit_transform(Z)

    # orient PHATE1 to increase with PSI for readability
    if spearmanr(Y[:, 0], tgt).statistic < 0:
        Y[:, 0] = -Y[:, 0]

    # drop stray PHATE outliers so the view zooms to the manifold (applied to
    # all arrays so every panel stays point-aligned)
    keep = drop_outliers(Y)
    n_drop = int((~keep).sum())
    Y, tgt, pred, err, neuron = Y[keep], tgt[keep], pred[keep], err[keep], neuron[keep]
    reads, intron, exon, glen = reads[keep], intron[keep], exon[keep], glen[keep]
    print(f"[outliers] dropped {n_drop} of {len(keep)} points", flush=True)

    # ---- quantification ----
    def sp(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(spearmanr(a[m], b[m]).statistic) if m.sum() > 10 else float("nan")
    q = {
        "n": int(len(tgt)),
        "rho_PHATE1_PSI": round(sp(Y[:, 0], tgt), 3),
        "rho_PHATE1_error": round(sp(Y[:, 0], err), 3),
        "rho_PHATE1_exonlen": round(sp(Y[:, 0], exon), 3),
        "rho_PHATE1_intronlen": round(sp(Y[:, 0], intron), 3),
        "rho_PHATE1_readdepth": round(sp(Y[:, 0], reads), 3) if cfg["has_reads"] else None,
    }
    # cell-type structure on the 2D manifold
    try:
        q["celltype_silhouette"] = round(float(silhouette_score(Y, neuron)), 3)
        knn = KNeighborsClassifier(n_neighbors=15)
        acc = cross_val_score(knn, Y, neuron, cv=3).mean()
        q["celltype_knn_acc"] = round(float(acc), 3)
        q["celltype_chance"] = round(1.0 / len(set(neuron)), 3)
    except Exception as e:
        q["celltype_error"] = str(e)
    summary[name] = q
    print(json.dumps(q, indent=2), flush=True)

    # ---- figure ----
    # PSI: sequential viridis, fixed 0..1 (both extremes visible).
    psi_norm = lambda: Normalize(vmin=0.0, vmax=1.0)
    # SIGNED error (pred − truth): diverging blue–white–red, white pinned at 0
    # (symmetric range) so blue = under-prediction, red = over-prediction.
    signed = pred - tgt
    sm = float(np.quantile(np.abs(signed), 0.99)) or 1.0
    err_norm = Normalize(vmin=-sm, vmax=sm)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    scatter(axes[0, 0], Y, tgt, "Ground-truth PSI", "viridis", "PSI", norm=psi_norm())
    scatter(axes[0, 1], Y, pred, "Predicted PSI", "viridis", "PSI", norm=psi_norm())
    scatter(axes[0, 2], Y, signed, "Prediction error (pred − truth)", "RdBu_r", "pred − truth", norm=err_norm)
    n_ct = len(set(neuron))
    scatter(axes[1, 0], Y, neuron, f"Cell type ({cfg['ctype']}, n={n_ct})", None,
            categorical=True, legend=(n_ct <= 12))
    scatter(axes[1, 1], Y, np.log10(exon + 1), "Cassette-exon length (log10 bp)", "cividis", "log10 bp")
    if cfg["has_reads"]:
        scatter(axes[1, 2], Y, np.log10(reads + 1), "Read depth (log10 reads)", "magma", "log10 reads")
    else:
        scatter(axes[1, 2], Y, np.log10(glen + 1), "Gene length (log10 bp)", "magma", "log10 bp")

    sub = (f"PHATE of fused latent (n={q['n']}).  "
           f"|ρ(PHATE1, PSI)|={abs(q['rho_PHATE1_PSI']):.2f}   "
           f"cell-type silhouette={q.get('celltype_silhouette')}  "
           f"(kNN acc {q.get('celltype_knn_acc')} vs chance {q.get('celltype_chance')})")
    fig.suptitle(cfg["title"] + "\n" + sub, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"phate_{name}.png")
    plt.close(fig)
    print(f"wrote {OUT}/phate_{name}.png", flush=True)

with open(OUT / "phate_quantification.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nwrote", OUT / "phate_quantification.json")
