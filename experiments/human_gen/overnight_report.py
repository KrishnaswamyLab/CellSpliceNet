"""Collect the Nov-24 overnight runs into <out>/human_gen/overnight_report.txt.
  1. tissue side (5 folds): lookup / low-rank tissue-level / low-rank sample-level, held-out tissues
  2. exon arms (5 folds): feat / conv / conv+feat / emb / emb+feat x tissue side, quadrants Q1-Q3 + exon-level ranking
  3. stage A (CSN without expression on event means): validation curve
  4. stage-B delta head on CSN embeddings (ASCOT-level, gene_x_cell split)
Env: TRAIN_LOGS / EMB_LOGS = globs of the stage-A training / embedding job logs (default logs/hg_stageA_*.out,
     logs/hg_emb_*.out; launch_stage_a.sh names the jobs so).
"""
import glob, re, os
import pandas as pd

from experiments.human_gen.common import out, out_dir

T1 = str(out("track1")) + "/"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
def agg(df, cols):
    g = df.groupby(["split", "method"])[cols]
    m = g.mean().round(3); s = g.std().round(3)
    t = m.astype(str) + " ± " + s.astype(str); t["folds"] = g.size(); return t

lines = []
lines.append("== 1. Tissue side, held-out tissues (5 folds, seed 0) ==")
f = glob.glob(T1 + "track1_results_seed0_fold?.csv")
if f:
    r = pd.concat(map(pd.read_csv, f)); r = r[r.split == "test"]
    lines.append(agg(r, ["within_all", "within_sd>=0.10", "within_sd>=0.20"]).to_string())

lines.append("\n== 2. Exon arms (5 folds): within-event per quadrant; Q2_between = exon-level ranking of mean PSI ==")
f = glob.glob(T1 + "track1_results_seed0_fold?_arms*.csv") + glob.glob(T1 + "track1_results_seed0_fold?.csv")
if f:
    r = pd.concat(map(pd.read_csv, f))
    r = r[r.method.str.startswith(("arm_", "BASE_", "featlowrank"))]
    r = r.drop_duplicates(["method", "split", "fold"])
    for q in ("Q1_new_tissue", "Q2_new_exon", "Q3_both_new", "Q2_between"):
        x = r[r.split == q]
        if len(x): lines.append(f"-- {q}\n" + agg(x, ["within_all", "within_sd>=0.10", "within_sd>=0.20"] if q != "Q2_between" else ["within_all"]).to_string())

lines.append("\n== 3. Stage A (csn_noexpr, event-mean targets, combo_em): valid MSE curve (constant predictor ~0.199) ==")
for lg in sorted(glob.glob(os.environ.get("TRAIN_LOGS", "logs/hg_stageA_*.out"))):
    v = re.findall(r"step\s+(\d+) \| train loss \(avg-since-last\): ([\d.]+) \| valid loss: ([\d.]+)", open(lg).read())
    lines.append(os.path.basename(lg) + ": " + " ".join(f"{s}:{vl}" for s, _, vl in v[::3]) + (f" ... last {v[-1][0]}:{v[-1][2]}" if v else ""))
for lg in glob.glob(os.environ.get("EMB_LOGS", "logs/hg_emb_*.out")):
    lines += [l.strip() for l in open(lg) if l.startswith("[emb]") and "rho" in l]

lines.append("\n== 4. Stage-B delta head on CSN embeddings (ASCOT-level, gene_x_cell split) ==")
p = str(out("delta_head")) + "/summary.txt"
lines.append(open(p).read() if os.path.exists(p) else "(not finished)")
txt = "\n".join(lines) + "\n"
open(out_dir("human_gen") / "overnight_report.txt", "w").write(txt); print(txt)
