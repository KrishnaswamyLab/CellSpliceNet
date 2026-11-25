"""Reconstruct human (GTEx) read depth per (event, tissue) from raw Zenodo junction counts.

ASCOT PSI carries no read counts, so the processed CSVs have nb_reads=1 (placeholder).
The junction counts ASCOT was derived from are local (data/zendo/exported_counts/<dir>/k_j_counts.tsv.gz).

For each cassette event (ASCOT exon_location + exon_boundary, validated empirically):
    upstream-inclusion junction = (chr, boundary_start, exon_start - 1)
    downstream-inclusion junction = (chr, exon_end + 1, boundary_end)
    exclusion junction           = (chr, boundary_start, boundary_end)
depth_total(event,tissue) = sum over the tissue's samples of (incl_up + incl_dn + excl) reads.
We also save n_samples and depth_per_sample = depth_total / n_samples.

Output -> outputs/reviewer_phate/descriptives/human_read_depth.csv(.gz)
Use --test to run only the first tissue and print match stats.
"""
import sys, pathlib, glob
import numpy as np
import pandas as pd

from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT  # folder prefix -> ASCOT/processed tissue name
from experiments.common.paths import HUMAN, gtex_counts_dir, out_dir, split_csv

ZENDO = gtex_counts_dir()
ASCOT_F = HUMAN / "raw/ascot/gtex_psi.csv.gz"
OUT = out_dir("reviewer", "descriptives")
TEST = "--test" in sys.argv

# ---- events we actually have ----
proc = pd.concat([pd.read_csv(split_csv("gene", s), usecols=["event_id", "neuron"])
                  for s in ["train", "valid", "test"]], ignore_index=True)
have_events = set(proc.event_id.unique())
have_tissues = set(proc.neuron.unique())
print(f"processed: {len(proc):,} rows | {len(have_events):,} events | {len(have_tissues)} tissues")

# ---- ASCOT coords -> the 3 junctions per event ----
asc = pd.read_csv(ASCOT_F, usecols=["exon_id", "exon_location", "exon_boundary"])
asc = asc[asc.exon_id.isin(have_events)].copy()


def parse(loc):
    c, r = loc.split(":"); s, e = r.split("-"); return c, int(s), int(e)


rows = []
for _, r in asc.iterrows():
    c, es, ee = parse(r.exon_location)
    c2, bs, be = parse(r.exon_boundary)
    rows.append((r.exon_id, c, es, ee, bs, be))
ev = pd.DataFrame(rows, columns=["event_id", "chr", "es", "ee", "bs", "be"])
# junction keys
ev["k_up"] = list(zip(ev.chr, ev.bs, ev.es - 1))
ev["k_dn"] = list(zip(ev.chr, ev.ee + 1, ev.be))
ev["k_ex"] = list(zip(ev.chr, ev.bs, ev.be))
print(f"ascot events with coords in our set: {len(ev):,}")

folders = sorted(p.name for p in ZENDO.iterdir() if p.is_dir())
if TEST:
    folders = folders[:1]

out_parts = []
for fi, folder in enumerate(folders, 1):
    prefix = folder.split("--")[0]
    tissue = FOLDER_TO_ASCOT.get(prefix)
    if tissue is None or tissue not in have_tissues:
        print(f"[{fi}/{len(folders)}] SKIP {folder} (tissue '{tissue}' not in set)")
        continue
    kj = pd.read_csv(ZENDO / folder / "k_j_counts.tsv.gz", sep="\t")
    sample_cols = [c for c in kj.columns if c not in ("seqnames", "start", "end", "strand")]
    n_samp = len(sample_cols)
    total = kj[sample_cols].sum(axis=1).to_numpy()
    # lookup: (chr,start,end) -> total reads
    lut = dict(zip(zip(kj.seqnames, kj.start, kj.end), total))

    def lk(keys):
        return np.array([lut.get(k, 0) for k in keys], dtype=np.int64)

    depth = lk(ev.k_up) + lk(ev.k_dn) + lk(ev.k_ex)
    matched = np.array([(k in lut) or (u in lut) or (x in lut)
                        for k, u, x in zip(ev.k_up, ev.k_dn, ev.k_ex)])
    out_parts.append(pd.DataFrame({"event_id": ev.event_id.values, "neuron": tissue,
                                   "depth_total": depth, "n_samples": n_samp}))
    print(f"[{fi}/{len(folders)}] {tissue:32s} n_samp={n_samp:3d} "
          f"events_with_any_junction={matched.mean()*100:5.1f}%  "
          f"median_depth_total={np.median(depth[depth>0]) if (depth>0).any() else 0:.0f}")

res = pd.concat(out_parts, ignore_index=True)
res["depth_per_sample"] = res.depth_total / res.n_samples
# restrict to (event,tissue) pairs actually in processed data
proc_keys = set(zip(proc.event_id, proc.neuron))
res = res[[(e, t) in proc_keys for e, t in zip(res.event_id, res.neuron)]].reset_index(drop=True)
print(f"\nfinal rows (in processed set): {len(res):,}")
print(f"nonzero depth: {(res.depth_total>0).mean()*100:.1f}%  "
      f"median per-sample (nonzero): {res.depth_per_sample[res.depth_total>0].median():.2f}")

fn = OUT / ("human_read_depth_TEST.csv" if TEST else "human_read_depth.csv.gz")
res.to_csv(fn, index=False)
print(f"wrote {fn}")
