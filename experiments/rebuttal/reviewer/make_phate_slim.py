"""Slim PHATE recoloring: error + cell type only (reviewer response).

Reuses the same fused-latent PHATE as make_phate.py (same StandardScaler +
PHATE(random_state=0) + PHATE1 orientation + MAD outlier drop, so layouts match
the full figures), but renders a compact two-wide panel (signed error | cell type)
for the human GTEx and worm event-based models only.

Run under `mioflow` (has phate). Outputs -> outputs/reviewer_phate/.
"""
import pathlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import phate
from experiments.common.paths import HUMAN, gtex_counts_dir, out_dir, split_csv

OUT = out_dir("reviewer")

DATASETS = {
    "worm_eventbased": {
        "title": "Worm CellSpliceNet — event-disjoint split",
        "ctype": "neuron type",
    },
    "human_gtex": {
        "title": "Human CellSpliceNet — GTEx",
        "ctype": "tissue",
    },
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


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
    med = np.median(Y, axis=0)
    d = np.linalg.norm(Y - med, axis=1)
    mad = np.median(np.abs(d - np.median(d))) or 1.0
    z = 0.6745 * (d - np.median(d)) / mad
    return z < thresh


for name, cfg in DATASETS.items():
    print(f"\n=== {name} ===", flush=True)
    zrep = np.load(OUT / f"{name}_zrep.npy")
    meta = pd.read_csv(OUT / f"{name}_meta.csv")
    tgt, pred = meta["target"].to_numpy(), meta["pred"].to_numpy()
    neuron = meta["neuron"].astype(str).to_numpy()

    Z = StandardScaler().fit_transform(zrep)
    op = phate.PHATE(n_components=2, n_jobs=-1, verbose=0, random_state=0)
    Y = op.fit_transform(Z)

    # orient PHATE1 to increase with PSI for readability
    if spearmanr(Y[:, 0], tgt).statistic < 0:
        Y[:, 0] = -Y[:, 0]

    keep = drop_outliers(Y)
    Y, tgt, pred, neuron = Y[keep], tgt[keep], pred[keep], neuron[keep]
    n = int(len(tgt))

    # SIGNED error (pred − truth): diverging, white pinned at 0
    signed = pred - tgt
    sm = float(np.quantile(np.abs(signed), 0.99)) or 1.0
    err_norm = Normalize(vmin=-sm, vmax=sm)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    scatter(axes[0], Y, signed, "Prediction error (pred − truth)", "RdBu_r",
            "pred − truth", norm=err_norm)
    n_ct = len(set(neuron))
    scatter(axes[1], Y, neuron, f"Cell type ({cfg['ctype']}, n={n_ct})", None,
            categorical=True, legend=(n_ct <= 12))

    fig.suptitle(cfg["title"] + f"   (PHATE of fused latent, n={n})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / f"phate_{name}_slim.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
