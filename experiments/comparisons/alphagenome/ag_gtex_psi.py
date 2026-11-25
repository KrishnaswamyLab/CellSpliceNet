"""AlphaGenome (released model, via API) native-splicing baseline on our GTEx test set.

Reviewer #2 asks for the released pretrained model's NATIVE splicing outputs, aggregated to
an exon-level inclusion value. For each cassette exon we request SPLICE_JUNCTIONS for the
GTEx tissues in our test set and compute

    PSI_hat = mean(incl_up, incl_dn) / (mean(incl_up, incl_dn) + excl)

from the predicted junction counts (AlphaGenome's own PSI5/PSI3 ratios are also count
ratios; absolute scale cancels). Junctions follow the convention validated for the GTEx
read-depth reconstruction (ASCOT 1-based inclusive coords):
    upstream inclusion  = intron [bs, es-1]
    downstream inclusion = intron [ee+1, be]
    exclusion            = intron [bs, be]
AlphaGenome intervals are 0-based; the start/end offset is checked in --pilot mode.

CAVEAT for interpretation: AlphaGenome's splicing heads were trained on GTEx RNA-seq from all
individuals, and the released 'all_folds' weights saw every genomic interval, so this is an
in-sample upper bound, not a held-out comparison.

Usage (API key in env, never on the command line):
  export ALPHAGENOME_API_KEY=...            # or: export ALPHAGENOME_API_KEY=$(cat ~/.alphagenome_key)
  venv/bin/python ag_gtex_psi.py --pilot    # 10 events: metadata dump + coordinate check
  venv/bin/python ag_gtex_psi.py --n_events 1000
"""
import argparse, json, os, pathlib, re, sys, time
import numpy as np
import pandas as pd

from experiments.common.paths import HUMAN, out_dir, split_csv

ASCOT = HUMAN / "raw/ascot/gtex_psi.csv.gz"
# test rows whose events / tissues are scored (default: the gene split's test set; CSN_PREDS = a predictions file)
CSN_PREDS = os.environ.get("CSN_PREDS") or split_csv("gene", "test")
OUT = out_dir("comparisons", "alphagenome")

p = argparse.ArgumentParser()
p.add_argument("--pilot", action="store_true")
p.add_argument("--n_events", type=int, default=1000)
p.add_argument("--seq_len", type=int, default=2**17)          # 131,072 bp
p.add_argument("--offset_start", type=int, default=None)       # set from pilot
p.add_argument("--offset_end", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

from alphagenome.data import genome
from alphagenome.models.v1 import dna_client

key = os.environ.get("ALPHAGENOME_API_KEY")
if not key:
    sys.exit("set ALPHAGENOME_API_KEY in the environment")
client = dna_client.create(key)
OUT.mkdir(parents=True, exist_ok=True)

# ---- tissue mapping: our GTEx tissue names -> AlphaGenome splice-junction track curies ----
meta = client.output_metadata(dna_client.Organism.HOMO_SAPIENS).splice_junctions
meta.to_csv(OUT / "ag_splice_junction_metadata.csv", index=False)
csn = pd.read_csv(CSN_PREDS)
tissues = sorted(csn.neuron.unique())

# ASCOT tissue name -> GTEx tissue name as used in AlphaGenome's 'gtex' junction tracks.
ASCOT_TO_GTEX = {
    "Adrenal Gland": "Adrenal_Gland", "Amygdala - Brain": "Brain_Amygdala",
    "Anterior cingulate - Brain": "Brain_Anterior_cingulate_cortex_BA24",
    "Aorta - Artery": "Artery_Aorta", "Atrial Appendage - Heart": "Heart_Atrial_Appendage",
    "Caudate nucleus - Brain": "Brain_Caudate_basal_ganglia",
    "Cerebellar Hemisphere - Brain": "Brain_Cerebellar_Hemisphere", "Cerebellum - Brain": "Brain_Cerebellum",
    "Coronary - Artery": "Artery_Coronary", "Cortex - Brain": "Brain_Cortex", "Cortex - Kidney": "Kidney_Cortex",
    "EBV-xform lymphocytes - Cells": "Cells_EBV-transformed_lymphocytes",
    "Frontal Cortex - Brain": "Brain_Frontal_Cortex_BA9",
    "Gastroesoph. Junc. - Esophagus": "Esophagus_Gastroesophageal_Junction",
    "Hippocampus - Brain": "Brain_Hippocampus", "Hypothalamus - Brain": "Brain_Hypothalamus",
    "Ileum - Small Intestine": "Small_Intestine_Terminal_Ileum", "Left Ventricle - Heart": "Heart_Left_Ventricle",
    "Liver": "Liver", "Lung": "Lung", "Mammary Tissue - Breast": "Breast_Mammary_Tissue",
    "Minor Salivary Gland": "Minor_Salivary_Gland", "Mucosa - Esophagus": "Esophagus_Mucosa",
    "Muscularis - Esophagus": "Esophagus_Muscularis", "Not Sun Exposed - Skin": "Skin_Not_Sun_Exposed_Suprapubic",
    "Nucleus accumbens - Brain": "Brain_Nucleus_accumbens_basal_ganglia", "Ovary": "Ovary",
    "Pancreas": "Pancreas", "Pituitary": "Pituitary", "Prostate": "Prostate",
    "Putamen - Brain": "Brain_Putamen_basal_ganglia", "Sigmoid - Colon": "Colon_Sigmoid",
    "Skeletal - Muscle": "Muscle_Skeletal", "Spinal cord (C1) - Brain": "Brain_Spinal_cord_cervical_c-1",
    "Spleen": "Spleen", "Stomach": "Stomach", "Subcutaneous - Adipose": "Adipose_Subcutaneous",
    "Substantia nigra - Brain": "Brain_Substantia_nigra", "Sun Exposed (Lower leg) - Skin": "Skin_Sun_Exposed_Lower_leg",
    "Testis": "Testis", "Thyroid": "Thyroid", "Tibial - Artery": "Artery_Tibial", "Tibial - Nerve": "Nerve_Tibial",
    "Transverse - Colon": "Colon_Transverse", "Uterus": "Uterus", "Vagina": "Vagina",
    "Visceral (Omentum) - Adipose": "Adipose_Visceral_Omentum", "Whole Blood": "Whole_Blood",
    "Xform. fibroblasts - Cells": "Cells_Cultured_fibroblasts",
}
gt = meta[meta.data_source == "gtex"].set_index("gtex_tissue")
tmap = pd.DataFrame([dict(tissue=t, gtex_tissue=ASCOT_TO_GTEX.get(t)) for t in tissues])
tmap["curie"] = tmap.gtex_tissue.map(gt.ontology_curie)
tmap["track"] = tmap.gtex_tissue.map(gt.name)       # use ONLY the GTEx track (ENCODE tracks can share a curie)
tmap.to_csv(OUT / "tissue_map.csv", index=False)
missing = tmap[tmap.track.isna()].tissue.tolist()
print(f"[ag] tissue map: {tmap.track.notna().sum()}/{len(tmap)} mapped to GTEx tracks; unmapped: {missing}")
tmap = tmap.dropna(subset=["track"])
curies = sorted(set(tmap.curie))

# ---- events ----
asc = pd.read_csv(ASCOT, usecols=["exon_id", "exon_strand", "exon_location", "exon_boundary"])
asc = asc[~asc.exon_location.str.contains(r"\|", na=False)].set_index("exon_id")
ev = pd.Series(csn.event_id.unique())
ev = ev[ev.isin(asc.index)].sample(frac=1, random_state=a.seed)
ev = ev.iloc[: (10 if a.pilot else a.n_events)].tolist()

def parse(loc):
    c, se = loc.split(":"); s, e = se.split("-"); return c, int(s), int(e)

def junction_table(jd, strand):
    keys = {}
    for i, j in enumerate(jd.junctions):
        if j.strand == strand:
            keys[(j.start, j.end)] = i
    return keys

def track_cols(jd, track):
    return np.where(jd.metadata["name"].values == track)[0]

rows, diag = [], []
t0 = time.time()
for k, e in enumerate(ev):
    r = asc.loc[e]
    chrom, es, ee = parse(r.exon_location); _, bs, be = parse(r.exon_boundary)
    strand = r.exon_strand
    mid = (es + ee) // 2
    L = a.seq_len if (be - bs) < a.seq_len - 2000 else 2**20
    iv = genome.Interval(chrom, mid - L // 2, mid + L // 2)
    try:
        out = client.predict_interval(iv, requested_outputs=[dna_client.OutputType.SPLICE_JUNCTIONS],
                                      ontology_terms=curies)
    except Exception as ex:  # rate limit / transient
        print(f"[ag] {e}: {type(ex).__name__}: {ex}", flush=True); time.sleep(10); continue
    jd = out.splice_junctions
    jt = junction_table(jd, strand)
    # 1-based inclusive intron [s, t] -> 0-based half-open [s-1, t] plus a tested offset
    cands = {"up": (bs, es - 1), "dn": (ee + 1, be), "ex": (bs, be)}
    offs = [(a.offset_start, a.offset_end)] if a.offset_start is not None else \
           [(ds, de) for ds in (-2, -1, 0, 1) for de in (-1, 0, 1)]
    hit = {}
    for ds, de in offs:
        idx = {n: jt.get((s - 1 + ds, t + de)) for n, (s, t) in cands.items()}
        hit[(ds, de)] = idx
        diag.append(dict(event_id=e, ds=ds, de=de, n_found=sum(v is not None for v in idx.values())))
    if a.pilot:
        continue
    idx = hit[offs[0]]
    for _, tm in tmap.iterrows():
        cols = track_cols(jd, tm.track)
        if len(cols) == 0:
            continue
        val = lambda n: 0.0 if idx[n] is None else float(jd.values[idx[n], cols].mean())
        up, dn, ex_ = val("up"), val("dn"), val("ex")
        inc = (up + dn) / 2
        rows.append(dict(event_id=e, neuron=tm.tissue, ag_up=up, ag_dn=dn, ag_ex=ex_,
                         ag_psi=inc / (inc + ex_) if inc + ex_ > 0 else np.nan))
    if k % 25 == 0:
        print(f"[ag] {k+1}/{len(ev)} events ({time.time()-t0:.0f}s)", flush=True)

d = pd.DataFrame(diag)
if a.pilot:
    s = d.groupby(["ds", "de"]).n_found.agg(["mean", "sum"]).sort_values("sum", ascending=False)
    print("[ag] junction matches by (start, end) offset; 3 = all of up/dn/excl found\n", s.head(6))
    print(f"[ag] pilot done. Re-run with --offset_start/--offset_end set to the top row.")
    sys.exit(0)

res = pd.DataFrame(rows)
res.to_csv(OUT / "ag_gtex_psi.csv", index=False)
print(f"[ag] wrote {len(res)} rows / {res.event_id.nunique()} events -> {OUT/'ag_gtex_psi.csv'}")
