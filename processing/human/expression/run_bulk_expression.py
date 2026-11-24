#!/usr/bin/env python3
"""
GTEx bulk tissue → CellSpliceNet expression tensors (same algorithm as CellSpliceNet/pp/expression).

Per tissue:
  geneCounts.tsv.gz → CPM → log1p → MAGIC → mutual-information graph (499×499) →
  per-gene moments across samples → graph scattering → scatter_coeffs_<ASCOT tissue>.pt

Output tensor shape: (n_sf, 11, n_sf) with n_sf = number of splice-factor genes (default 499).

Requires: pandas, numpy, scipy, sklearn, torch, torch_geometric, torch_scatter, magic-impute
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_regression
from torch_geometric.data import Data
from tqdm import tqdm

# graph scattering shared with the worm expression and structure pipelines
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from processing.common import default_data_root  # noqa: E402
from processing.scatter import Scatter  # noqa: E402
from processing.human.expression.gencode_symbols import parse_gencode_gene_symbols  # noqa: E402
from processing.human.expression.gobp_genes import load_gobp_symbols  # noqa: E402
from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT  # noqa: E402

_MAGIC = None


def _magic_impute(df_genes_samples: pd.DataFrame) -> pd.DataFrame:
    """MAGIC expects rows = cells/samples, columns = genes (same as scanpy)."""
    global _MAGIC
    import magic

    if _MAGIC is None:
        _MAGIC = magic.MAGIC()
    imputed_samples_genes = _MAGIC.fit_transform(df_genes_samples.T)
    return imputed_samples_genes.T


def _cpm_log1p(counts: pd.DataFrame) -> pd.DataFrame:
    """Samples are columns, genes are rows. CPM per sample then log1p."""
    col_sums = counts.sum(axis=0)
    col_sums = col_sums.replace(0, np.nan)
    cpm = counts.div(col_sums, axis=1) * 1.0e6
    cpm = cpm.fillna(0.0)
    return np.log1p(cpm)


def _build_mi_matrix(df_genes_samples: pd.DataFrame) -> pd.DataFrame:
    """
    df: index=gene ENSG (or labels), columns=samples — same orientation as pp01 filtered output
    before pp02 split. Original pp03 transposes to samples × genes then MI on columns.
    """
    # genes × samples → samples × genes (columns = genes)
    g = df_genes_samples.T
    gene_labels = list(g.columns)
    n = len(gene_labels)
    mi_mat = pd.DataFrame(np.zeros((n, n)), index=gene_labels, columns=gene_labels)
    for i, col1 in enumerate(tqdm(gene_labels, desc="MI rows", leave=False)):
        for j, col2 in enumerate(gene_labels):
            if i == j:
                mi_mat.iloc[i, j] = 0.0
                continue
            mi = mutual_info_regression(g[[col1]], g[col2], random_state=0)
            mi_mat.iloc[i, j] = float(mi[0])
    return mi_mat


def _moments_df(df_genes_samples: pd.DataFrame) -> pd.DataFrame:
    """Mean / var / skew / kurtosis across samples (axis=1), matching pp02."""
    m = df_genes_samples.mean(axis=1)
    v = df_genes_samples.var(axis=1)
    sk = df_genes_samples.apply(skew, axis=1)
    ku = df_genes_samples.apply(kurtosis, axis=1)
    return pd.DataFrame({"mean": m, "variance": v, "skewness": sk, "kurtosis": ku})


def _build_graph_continuous(adj_mat: pd.DataFrame, moments_data: pd.DataFrame) -> Data:
    adj = torch.tensor(adj_mat.values, dtype=torch.float32)
    edge_index = np.vstack(np.triu(adj.numpy(), k=1).nonzero())
    edge_weight = adj[edge_index[0], edge_index[1]]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = edge_weight.clone().detach().to(torch.float32)
    node_features = torch.diag(torch.tensor(moments_data["mean"].values, dtype=torch.float32))
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)


def _scatter_graph(graph: Data, device: torch.device) -> torch.Tensor:
    in_channels = graph.x.shape[-1]
    n_nodes = graph.x.size(0)
    scattering = Scatter(in_channels, n_nodes).to(device)
    graph = graph.to(device)
    scattering.eval()
    with torch.no_grad():
        out = scattering(graph)
    return out.cpu()


def _load_counts_matrix(
    counts_tsv_gz: Path, ensg_ids: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Rows = genes (ensg_ids order), columns = numeric sample counts.
    Genes absent from this tissue file are zero-filled (not in quantification matrix).
    """
    df = pd.read_csv(counts_tsv_gz, sep="\t", compression="gzip")
    if "geneID" not in df.columns:
        raise ValueError(f"Expected geneID column in {counts_tsv_gz}")
    base = df["geneID"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    df = df.assign(_ensg_base=base)
    sample_cols = (
        df.drop(columns=["geneID", "_ensg_base"], errors="ignore")
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    if not sample_cols:
        raise RuntimeError(f"No numeric sample columns in {counts_tsv_gz}")

    sub = df[df["_ensg_base"].isin(ensg_ids)].copy()
    sub = sub.drop_duplicates(subset="_ensg_base", keep="first")
    sub = sub.set_index("_ensg_base")
    zero_filled: list[str] = []
    rows: list[pd.Series] = []
    for e in ensg_ids:
        if e in sub.index:
            row = sub.loc[e]
            vals = row.reindex(sample_cols).astype(np.float64)
        else:
            zero_filled.append(e)
            vals = pd.Series(0.0, index=sample_cols, dtype=np.float64)
        rows.append(vals)
    out = pd.DataFrame(rows, index=list(ensg_ids), columns=sample_cols)
    return out, zero_filled


def run_one_tissue(
    folder_name: str,
    exported_counts_root: Path,
    ensg_ordered: list[str],
    gtf_path: Path,
    out_dir: Path,
    device: str,
    skip_magic: bool = False,
) -> dict:
    ascot = FOLDER_TO_ASCOT[folder_name]
    tsv = exported_counts_root / f"{folder_name}--GRCh38--gencode29" / "geneCounts.tsv.gz"
    if not tsv.is_file():
        return {"folder": folder_name, "ascot": ascot, "status": "missing_file", "path": str(tsv)}

    counts, zero_filled = _load_counts_matrix(tsv, ensg_ordered)
    logged = _cpm_log1p(counts)

    if skip_magic:
        magic_df = logged
    else:
        magic_df = _magic_impute(logged)

    mi = _build_mi_matrix(magic_df)
    moments = _moments_df(magic_df)
    # align moments / mi to same index
    moments = moments.reindex(mi.index)
    graph = _build_graph_continuous(mi, moments)

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    coeffs = _scatter_graph(graph, dev)
    expected = (len(ensg_ordered), 11, len(ensg_ordered))
    if tuple(coeffs.shape) != expected:
        raise RuntimeError(f"Bad scatter shape {coeffs.shape}, expected {expected}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"scatter_coeffs_{ascot}.pt"
    torch.save(coeffs, out_pt)
    mi.to_csv(out_dir / f"mi_matrix__{folder_name}.csv")
    moments.to_csv(out_dir / f"moments__{folder_name}.csv")

    return {
        "folder": folder_name,
        "ascot": ascot,
        "status": "ok",
        "n_samples": int(counts.shape[1]),
        "shape": list(coeffs.shape),
        "out": str(out_pt),
        "zero_filled_ensg_in_counts": zero_filled,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="GTEx bulk → scatter_coeffs_<tissue>.pt")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root (default $CSN_DATA_ROOT or <repo>/dataset); reads <root>/human/raw",
    )
    p.add_argument("--gencode-gtf", type=Path, default=None, help="Override path to gencode .gtf.gz")
    p.add_argument("--gobp-grp", type=Path, default=None, help="Override path to GOBP .grp")
    p.add_argument(
        "--exported-counts",
        type=Path,
        default=None,
        help="Directory containing Adipose_-_Subcutaneous--GRCh38--gencode29/ ...",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for .pt and intermediates (default: <root>/human/processed/expression/scatter_493)",
    )
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--tissue", type=str, default=None, help="Single folder key, e.g. Liver")
    p.add_argument("--skip-magic", action="store_true", help="Debug: skip MAGIC step")
    p.add_argument(
        "--allow-missing-gobp",
        action="store_true",
        help="Drop GOBP symbols not found in GENCODE (v38 naming); see build_meta.json n_genes",
    )
    p.add_argument(
        "--init-only",
        action="store_true",
        help="Write exact_order + build_meta only (for dSQ parallel setup).",
    )
    p.add_argument(
        "--from-existing-order",
        action="store_true",
        help="Load ENSG list from out_dir/exact_order_of_splice_factors.csv (parallel tissue jobs).",
    )
    args = p.parse_args()

    raw = (args.data_root or default_data_root()) / "human" / "raw"
    gobp = args.gobp_grp or (raw / "GOBP_RNA_SPLICING.v2026.1.Hs.grp")
    exported = args.exported_counts or (raw / "gtex_counts" / "exported_counts")
    out_dir = args.out_dir or (raw.parent / "processed" / "expression" / "scatter_493")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not exported.is_dir():
        raise SystemExit(f"Missing exported_counts: {exported}")

    gtf = args.gencode_gtf or (raw / "hg38" / "gencode.v38.annotation.gtf.gz")
    unresolved: list[str] = []
    sym2ensg: dict[str, str] = {}

    if args.from_existing_order:
        order_path = out_dir / "exact_order_of_splice_factors.csv"
        if not order_path.is_file():
            raise SystemExit(f"Missing {order_path}; run --init-only first.")
        ensg_ordered = [
            ln.strip() for ln in order_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if not ensg_ordered:
            raise SystemExit(f"Empty gene order file: {order_path}")
    else:
        if not gtf.is_file():
            raise SystemExit(f"Missing GTF: {gtf}")
        if not gobp.is_file():
            raise SystemExit(f"Missing GOBP grp: {gobp}")

        symbols = load_gobp_symbols(gobp)
        sym2ensg = parse_gencode_gene_symbols(str(gtf))
        ensg_ordered = []
        for s in symbols:
            e = sym2ensg.get(s)
            if e is None:
                unresolved.append(s)
            else:
                ensg_ordered.append(e)
        if unresolved:
            msg = (
                f"{len(unresolved)} GOBP symbols missing from GENCODE v38 gene map "
                f"(often newer symbols / readthroughs). Examples: {unresolved[:20]}"
            )
            if not args.allow_missing_gobp:
                raise SystemExit(msg + "\nUse --allow-missing-gobp to drop them and continue.")
            print("WARNING:", msg)

        (out_dir / "exact_order_of_splice_factors.csv").write_text(
            "\n".join(ensg_ordered) + "\n", encoding="utf-8"
        )
        with open(out_dir / "exact_order_of_splice_factors.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["order", "symbol", "ensg_id"])
            sym_for_ensg = {v: k for k, v in sym2ensg.items()}
            for i, e in enumerate(ensg_ordered):
                w.writerow([i, sym_for_ensg.get(e, "?"), e])

        meta = {
            "n_genes": len(ensg_ordered),
            "dropped_gobp_symbols": unresolved,
            "gtf": str(gtf),
            "gobp": str(gobp),
            "exported_counts": str(exported),
        }
        (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.init_only:
        if args.from_existing_order:
            raise SystemExit("Use only one of --init-only and --from-existing-order.")
        print(f"Wrote gene order and build_meta to {out_dir} (n={len(ensg_ordered)})")
        return

    folders = sorted(FOLDER_TO_ASCOT.keys())
    if args.tissue:
        if args.tissue not in FOLDER_TO_ASCOT:
            raise SystemExit(f"Unknown tissue key {args.tissue!r}. Keys look like: Liver, Brain_-_Cortex, ...")
        folders = [args.tissue]
    elif args.from_existing_order:
        raise SystemExit("--from-existing-order requires --tissue (one job per tissue).")

    results = []
    for folder in tqdm(folders, desc="tissues"):
        r = run_one_tissue(
            folder,
            exported,
            ensg_ordered,
            gtf,
            out_dir,
            args.device,
            skip_magic=args.skip_magic,
        )
        results.append(r)
        if r["status"] != "ok":
            tqdm.write(str(r))
        if args.from_existing_order and args.tissue:
            piece = out_dir / f"run_piece_{folder}.json"
            piece.write_text(json.dumps(r, indent=2), encoding="utf-8")

    if not (args.from_existing_order and args.tissue):
        (out_dir / "run_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Done. {ok}/{len(results)} tissues written to {out_dir}")


if __name__ == "__main__":
    main()
