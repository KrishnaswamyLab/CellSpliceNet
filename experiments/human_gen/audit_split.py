"""Audit of the human (GTEx/ASCOT) gene-disjoint split: is unseen-gene performance generalization or homology?

Outputs (<out>/human_gen/audit/):
  event_features.csv      per-event sequence features (lengths, GC, MaxEnt 5'/3' scores, chromosome)
  split_stats.txt         rows/events/genes/tissues/chromosomes per split, within-event variance share
  blast_test_vs_train.tsv best BLAST hit of each test exon (+-100 nt flanks) among training exons
  audit_results.csv       CSN (3 models) and baselines on the test set: pooled / between-event / within-event,
                          overall and stratified by similarity to training
CSN predictions: <out>/human_full/gtex_fulltest/<tag>/preds_<tag>.csv for tag = original, seed1, seed2
(experiments/human_full; FULLTEST_PREDS overrides the folder); tags without a file are skipped.
Needs maxentpy, and BLAST+ (module load BLAST+/2.15.0-gompi-2022b, or BLAST_MODULE='' if on PATH).
Baselines (all fit on training rows only):
  feat_gbm      gradient boosting on 14 sequence features -> event-mean PSI (+ tissue offset)
  kmer_ridge    ridge on 1-4-mer counts of exon / 100 nt upstream / 100 nt downstream
  nn_blast      PSI of the most similar training exon (BLAST bitscore); global mean if no hit
  tissue_offset training-set tissue mean deviation from event mean (within-event reference)
"""
import os, sys, subprocess, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from maxentpy import maxent
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from experiments.common.paths import out_dir
from experiments.human_gen.common import ASCOT_PSI, GENOME, out, read_split
from processing.human.genome import Genome

OUT = str(out("audit")) + "/"
FLANK = 300
log = open(OUT + "split_stats.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

# ---------------- data ----------------
sp = {k: read_split("gene", k, usecols=["gene_id", "event_id", "PSI", "neuron"]) for k in ("train", "valid", "test")}
asc = pd.read_csv(ASCOT_PSI, usecols=["exon_id", "exon_strand", "exon_length", "exon_location", "exon_boundary"],
                  encoding="utf-8-sig").rename(columns={"exon_id": "event_id"})
ev = pd.concat([d[["event_id", "gene_id"]].drop_duplicates() for d in sp.values()]).drop_duplicates("event_id").merge(asc, on="event_id", how="left")
say("events with ASCOT coordinates:", ev.exon_location.notna().sum(), "/", len(ev))

# ---------------- sequence features ----------------
fa = Genome(GENOME)
comp = str.maketrans("ACGTN", "TGCAN")
def seq(ch, a, b, strand):  # 1-based inclusive
    s = fa.fetch(ch, a - 1, b)
    return s.translate(comp)[::-1] if strand == "-" else s
def gc(s): return (s.count("G") + s.count("C")) / max(len(s), 1)
M3, M5 = maxent.load_matrix3(), maxent.load_matrix5()
rows, fasta_lines = [], {}
for r in ev.itertuples():
    if not isinstance(r.exon_location, str): continue
    ch, xy = r.exon_location.split(":"); a, b = map(int, xy.split("-"))
    _, bxy = r.exon_boundary.split(":"); ba, bb = map(int, bxy.split("-"))
    up_len, dn_len = (a - ba, bb - b) if r.exon_strand == "+" else (bb - b, a - ba)
    S = seq(ch, a - FLANK, b + FLANK, r.exon_strand); L = b - a + 1
    ex, up, dn = S[FLANK:FLANK + L], S[FLANK - 100:FLANK], S[FLANK + L:FLANK + L + 100]
    acc, don = S[FLANK - 20:FLANK + 3], S[FLANK + L - 3:FLANK + L + 6]
    try: m3 = maxent.score3(acc, matrix=M3) if len(acc) == 23 and "N" not in acc else np.nan
    except Exception: m3 = np.nan
    try: m5 = maxent.score5(don, matrix=M5) if len(don) == 9 and "N" not in don else np.nan
    except Exception: m5 = np.nan
    rows.append(dict(event_id=r.event_id, chrom=ch, exon_len=L, up_intron=up_len, dn_intron=dn_len, log_exon=np.log10(L),
                     log_up=np.log10(max(up_len, 1)), log_dn=np.log10(max(dn_len, 1)), len_mod3=L % 3,
                     gc_exon=gc(ex), gc_up=gc(up), gc_dn=gc(dn), maxent3=m3, maxent5=m5,
                     ag_up=up[-30:].count("AG"), ex_seq=ex, up_seq=up, dn_seq=dn))
    fasta_lines[r.event_id] = up + ex + dn
feat = pd.DataFrame(rows)
feat.drop(columns=["ex_seq", "up_seq", "dn_seq"]).to_csv(OUT + "event_features.csv", index=False)
FCOLS = ["log_exon", "log_up", "log_dn", "len_mod3", "gc_exon", "gc_up", "gc_dn", "maxent3", "maxent5", "ag_up"]

# ---------------- split stats ----------------
chrom = feat.set_index("event_id").chrom
for k, d in sp.items():
    c = d.event_id.map(chrom).value_counts()
    say(f"{k}: rows={len(d):,} events={d.event_id.nunique():,} genes={d.gene_id.nunique():,} tissues={d.neuron.nunique()} "
        f"chroms={len(c)} top={dict(c.head(4))}")
say("gene overlap train/test:", len(set(sp['train'].gene_id) & set(sp['test'].gene_id)),
    " event overlap:", len(set(sp['train'].event_id) & set(sp['test'].event_id)))
te = sp["test"]
m = te.groupby("event_id").PSI.transform("mean")
say(f"test PSI variance share within events: {((te.PSI - m) ** 2).sum() / ((te.PSI - te.PSI.mean()) ** 2).sum():.3f}")
em = te.groupby("event_id").PSI.mean()
say(f"test events: mean PSI>0.95 {np.mean(em > .95):.2f}  <0.05 {np.mean(em < .05):.2f}  intermediate {np.mean((em >= .05) & (em <= .95)):.2f}")

# ---------------- BLAST similarity ----------------
tr_ev, te_ev = set(sp["train"].event_id), set(te.event_id)
with open(OUT + "train.fa", "w") as f:
    for e in tr_ev:
        if e in fasta_lines: f.write(f">{e}\n{fasta_lines[e]}\n")
with open(OUT + "test.fa", "w") as f:
    for e in te_ev:
        if e in fasta_lines: f.write(f">{e}\n{fasta_lines[e]}\n")
mod = os.environ.get("BLAST_MODULE", "BLAST+/2.15.0-gompi-2022b")
sh = ((f"module load {mod} && " if mod else "") + f"makeblastdb -in {OUT}train.fa -dbtype nucl -out {OUT}traindb >/dev/null && "
      f"blastn -task blastn -query {OUT}test.fa -db {OUT}traindb -evalue 1e-5 -outfmt '6 qseqid sseqid pident length bitscore evalue' "
      f"-max_target_seqs 5 -num_threads {os.environ.get('SLURM_CPUS_PER_TASK', '4')} -dust no > {OUT}blast_raw.tsv")
subprocess.run(["bash", "-lc", sh], check=True)
bl = pd.read_csv(OUT + "blast_raw.tsv", sep="\t", names=["event_id", "hit", "pident", "alen", "bits", "evalue"])
bl = bl.sort_values("bits", ascending=False).drop_duplicates("event_id")
bl.to_csv(OUT + "blast_test_vs_train.tsv", sep="\t", index=False)
say(f"test events with a BLAST hit in train (e<1e-5): {len(bl)} / {len(te_ev)}; pident>=90 & alen>=100: {((bl.pident >= 90) & (bl.alen >= 100)).sum()}")

# similarity bins
bins = pd.Series("none", index=sorted(te_ev))
b = bl.set_index("event_id")
bins[b.index[(b.bits < 100)]] = "weak(<100 bits)"
bins[b.index[(b.bits >= 100) & (b.bits < 300)]] = "moderate(100-300)"
bins[b.index[(b.bits >= 300)]] = "strong(>=300)"
say("similarity bins:", bins.value_counts().to_dict())

# ---------------- baselines ----------------
tr = sp["train"]
tr_em = tr.groupby("event_id").PSI.mean()
F = feat.set_index("event_id")
Xtr = F.loc[F.index.intersection(tr_em.index), FCOLS]; ytr = tr_em.loc[Xtr.index]
gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
def kmers(s, ks=(1, 2, 3, 4)):
    v = []
    for k in ks:
        idx = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=k))}
        c = np.zeros(len(idx))
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k]);
            if j is not None: c[j] += 1
        v.append(c / max(len(s) - k + 1, 1))
    return np.concatenate(v)
KM = {r.event_id: np.concatenate([kmers(r.ex_seq), kmers(r.up_seq), kmers(r.dn_seq)]) for r in feat.itertuples()}
ktr = [e for e in Xtr.index if e in KM]
lo = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
ridge = Ridge(alpha=1.0).fit(np.stack([KM[e] for e in ktr]), lo(tr_em.loc[ktr].values))
tr["dev"] = tr.PSI - tr.groupby("event_id").PSI.transform("mean")
toff = tr.groupby("neuron").dev.mean()

te = te.copy()
te_ids = te.event_id.unique()
Fte = F.reindex(te_ids)
te["feat_gbm"] = te.event_id.map(pd.Series(gbm.predict(Fte[FCOLS]), index=te_ids))
te["kmer_ridge"] = te.event_id.map(pd.Series(1 / (1 + np.exp(-ridge.predict(np.stack([KM.get(e, np.zeros(len(next(iter(KM.values()))))) for e in te_ids])))), index=te_ids))
nn = b.hit.map(tr_em)
te["nn_blast"] = te.event_id.map(nn).fillna(tr_em.mean())
te["tissue_offset"] = te.neuron.map(toff).fillna(0)
te["feat_gbm_plus_off"] = te.feat_gbm + te.tissue_offset

# ---------------- CSN predictions ----------------
FT = os.environ.get("FULLTEST_PREDS", str(out_dir("human_full", "gtex_fulltest"))) + "/"
csn = []
TAGS = [t for t in ("original", "seed1", "seed2") if os.path.exists(FT + f"{t}/preds_{t}.csv")]
for tag in TAGS:
    d = pd.read_csv(FT + f"{tag}/preds_{tag}.csv").rename(columns={"psi_pred": f"csn_{tag}", "psi_target": "t_" + tag})
    csn.append(d[["event_id", "neuron", f"csn_{tag}"]])
for d in csn: te = te.merge(d, on=["event_id", "neuron"], how="left")
if TAGS: te["csn_mean"] = te[[f"csn_{t}" for t in TAGS]].mean(1)
te["sim_bin"] = te.event_id.map(bins)

def within(d, col):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); bb = d[col] - d.groupby("event_id")[col].transform("mean")
    return spearmanr(a, bb).correlation if bb.std() > 1e-9 else 0.0
def metrics(d, col):
    d = d.dropna(subset=[col]); em_ = d.groupby("event_id")[["PSI", col]].mean()
    return dict(n=len(d), events=d.event_id.nunique(), pooled=spearmanr(d.PSI, d[col]).correlation,
                between=spearmanr(em_.PSI, em_[col]).correlation, within=within(d, col), pearson=pearsonr(d.PSI, d[col])[0])
COLS = [f"csn_{t}" for t in TAGS] + (["csn_mean"] if TAGS else []) + ["feat_gbm", "kmer_ridge", "nn_blast", "tissue_offset", "feat_gbm_plus_off"]
res = []
for grp, d in [("all", te)] + list(te.groupby("sim_bin")):
    for c in COLS:
        res.append(dict(subset=grp, model=c, **metrics(d, c)))
res = pd.DataFrame(res).round(3); res.to_csv(OUT + "audit_results.csv", index=False)
pd.set_option("display.width", 200); say(res.to_string())
te.drop(columns=[]).to_csv(OUT + "test_with_baselines.csv.gz", index=False)
say("done")
