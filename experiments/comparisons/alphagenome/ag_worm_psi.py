"""Zero-shot AlphaGenome on C. elegans skipped exons (never trained on worm splicing data).

For each SE event we decode the stored whole-gene pre-mRNA (sense strand, tokens
A=1 G=2 U=3 C=4 X=5), centre it in an N-padded 131,072-bp input, call predict_sequence
for SPLICE_JUNCTIONS on the GTEx tracks, and compute
    PSI_hat = mean(incl_up, incl_dn) / (mean(incl_up, incl_dn) + excl)
from counts averaged over (a) all 49 GTEx tissue tracks and (b) the brain tracks only.
AlphaGenome has no worm neurons, so this is one PSI per event (a between-event predictor).

Gene-relative coordinate convention is unknown -> --pilot searches a (start, end) offset
independently for each of the three junctions.

  export ALPHAGENOME_API_KEY=$(cat ~/.alphagenome_key)
  venv/bin/python ag_worm_psi.py --pilot
  venv/bin/python ag_worm_psi.py --off_up 0 0 --off_dn 0 0 --off_ex 0 0
"""
import argparse, itertools, os, pathlib, sys, time
import json
import numpy as np, pandas as pd

from experiments.common.paths import WORM, out_dir, split_csv

OUT = out_dir("comparisons", "alphagenome")
L = 2**17

p = argparse.ArgumentParser()
p.add_argument("--pilot", action="store_true")
for j in ("up", "dn", "ex"):
    p.add_argument(f"--off_{j}", type=int, nargs=2, default=None)
a = p.parse_args()

from alphagenome.models.v1 import dna_client
client = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])
meta = client.output_metadata(dna_client.Organism.HOMO_SAPIENS).splice_junctions
gt = meta[meta.data_source == "gtex"]
tracks_all = gt.name.tolist()
tracks_brain = gt[gt.gtex_tissue.str.startswith("Brain")].name.tolist()
curies = sorted(set(gt.ontology_curie))

# all SE events with coordinates (union of the row-level split; coordinates are per event)
frames = []
for s in ("train", "valid", "test"):
    frames.append(pd.read_csv(split_csv("event_sample1", s, "c_elegans")))
ev = (pd.concat(frames)[["event_id", "gene_id", "exon_start", "exon_end", "intron_start", "intron_end", "gene_length"]]
      .drop_duplicates("event_id").reset_index(drop=True))
# gene sequences exported from encoded_sequences.pth (tokens 0 PAD->N, 1 A, 2 G, 3 U->T, 4 C, 5 X->N)
if not (OUT / "worm_gene_seqs.json").exists():
    import torch
    enc = torch.load(WORM / "processed/encoded_sequences.pth", map_location="cpu", weights_only=False)
    tok = np.array(list("NAGTCN"))
    seqs = {g: "".join(tok[np.asarray(enc[g]).reshape(-1).astype(int)]) for g in ev.gene_id.unique() if g in enc}
    json.dump(seqs, open(OUT / "worm_gene_seqs.json", "w"))
seqs = json.load(open(OUT / "worm_gene_seqs.json"))
if a.pilot:
    ev = ev.sample(12, random_state=0)
print(f"[agw] events={len(ev)} gtex tracks={len(tracks_all)} brain tracks={len(tracks_brain)}", flush=True)

def build(row):
    s = seqs[row.gene_id]
    if len(s) <= L:
        left = (L - len(s)) // 2
        return "N" * left + s + "N" * (L - len(s) - left), left
    mid = (row.exon_start + row.exon_end) // 2
    st = int(min(max(0, mid - L // 2), len(s) - L))
    return s[st:st + L], -st

offs_grid = list(itertools.product(range(-2, 3), repeat=2))
rows, diag = [], []
t0 = time.time()
for k, r in enumerate(ev.itertuples()):
    if r.gene_id not in seqs:
        continue
    seq, shift = build(r)
    try:
        out = client.predict_sequence(sequence=seq, requested_outputs=[dna_client.OutputType.SPLICE_JUNCTIONS],
                                      ontology_terms=curies)
    except Exception as ex:
        print(f"[agw] {r.event_id}: {type(ex).__name__}: {ex}", flush=True); time.sleep(10); continue
    jd = out.splice_junctions
    plus = {(j.start, j.end): i for i, j in enumerate(jd.junctions) if j.strand == "+"}
    base = {"up": (r.intron_start, r.exon_start), "dn": (r.exon_end, r.intron_end), "ex": (r.intron_start, r.intron_end)}
    if a.pilot:
        for jn, (s0, e0) in base.items():
            for ds, de in offs_grid:
                diag.append(dict(event_id=r.event_id, junction=jn, ds=ds, de=de,
                                 found=(s0 + shift + ds, e0 + shift + de) in plus))
        continue
    cols_all = np.where(np.isin(jd.metadata["name"].values, tracks_all))[0]
    cols_brain = np.where(np.isin(jd.metadata["name"].values, tracks_brain))[0]
    rec = dict(event_id=r.event_id, gene_id=r.gene_id)
    for tag, cols in (("all", cols_all), ("brain", cols_brain)):
        v = {}
        for jn, (s0, e0) in base.items():
            ds, de = getattr(a, f"off_{jn}")
            i = plus.get((s0 + shift + ds, e0 + shift + de))
            v[jn] = 0.0 if i is None else float(jd.values[i, cols].mean())
            rec[f"found_{jn}"] = i is not None
        inc = (v["up"] + v["dn"]) / 2
        rec[f"ag_psi_{tag}"] = inc / (inc + v["ex"]) if inc + v["ex"] > 0 else np.nan
    rows.append(rec)
    if k % 50 == 0:
        print(f"[agw] {k+1}/{len(ev)} ({time.time()-t0:.0f}s)", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
if a.pilot:
    d = pd.DataFrame(diag)
    best = (d.groupby(["junction", "ds", "de"]).found.sum().reset_index()
              .sort_values(["junction", "found"], ascending=[True, False]).groupby("junction").head(2))
    print("[agw] best offsets per junction (found = #events of 12 matched):\n", best.to_string(index=False))
    sys.exit(0)
res = pd.DataFrame(rows)
res.to_csv(OUT / "ag_worm_psi.csv", index=False)
print(f"[agw] wrote {len(res)} events; all three junctions found: "
      f"{(res.found_up & res.found_dn & res.found_ex).mean():.1%} -> {OUT/'ag_worm_psi.csv'}")
