"""Pre-submission data-coverage check for a worm data config.

Verifies, on CPU in seconds-to-a-minute, that a config's split files are covered by
the sequence/SJ/spliceregion/coordinate assets a training run will need -- so a GPU
job is not queued only to die on a KeyError. Also reports the effect of the
dataloader's p_seq_len < max_prime_seq_len (25000 nt) filter, which silently drops
~14% of rows.

With --deep it additionally checks the structure pickle and the expression scatter
directory. csn_seqonly does not use either (see model/seqonly_baseline.py), so a
--deep failure only matters for full-model runs, or tells you to set
SEQONLY_SYNTH_SHAPES=1.

Usage:
  python experiments/worm_ablations/preflight.py <split name | data_config.ini> [--deep] [--max-seq-len 25000]
"""
import argparse, os, sys
from dataclasses import asdict
import pandas as pd

from experiments.worm_ablations.common import split_ini
from utils.paths import load_dataset_bundle

def section(msg): print(f"\n=== {msg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=25000)
    ap.add_argument("--structure-pkl", default=None)
    ap.add_argument("--scatter-dir", default=None)
    a = ap.parse_args()

    a.config = str(split_ini(a.config))
    pf = {k: str(v) for k, v in asdict(load_dataset_bundle(a.config)).items()
          if v is not None and k not in ("config_file", "mean_vec_dir", "graph_metric_dir")}
    a.structure_pkl = a.structure_pkl or pf.pop("structure_data_root")   # --deep checks
    a.scatter_dir = a.scatter_dir or pf.pop("scatter_coeffs_dir")
    pf.pop("structure_data_root", None); pf.pop("scatter_coeffs_dir", None)
    ok = True

    section(f"file existence ({a.config})")
    for k in pf:
        p = pf[k]; e = os.path.exists(p)
        readable = e and os.access(p, os.R_OK)
        print(f"  {'OK ' if readable else 'FAIL'} {k:20s} {p}")
        ok &= readable
    if not ok:
        print("\n-> missing/unreadable inputs; fix before submitting."); return 1

    section("split sizes and the p_seq_len filter")
    parts = {}
    for s in ("train", "valid", "test"):
        d = pd.read_csv(pf[f"{s}_data_file"]); parts[s] = d
        kept = d[d.p_seq_len < a.max_seq_len]
        print(f"  {s:5s} rows {len(d):6d} -> usable {len(kept):6d} ({100*len(kept)/len(d):.1f}%)"
              f" | events {kept.event_id.nunique():4d} genes {kept.gene_id.nunique():4d}"
              f" neurons {kept.neuron.nunique():3d}")
    allrows = pd.concat(parts.values())
    allrows = allrows[allrows.p_seq_len < a.max_seq_len]
    events = set(allrows.event_id.astype(str)); genes = set(allrows.gene_id.astype(str))
    neurons = set(allrows.neuron.astype(str))

    section("group disjointness between train and test")
    for key in ("event_id", "gene_id", "neuron"):
        tr = set(parts["train"][key]); te = set(parts["test"][key])
        print(f"  {key:9s} shared train&test: {len(tr & te):5d}"
              f"   {'(held out)' if not (tr & te) else ''}")

    section("spliceregion / coordinate coverage")
    sr = pd.read_csv(pf["spliceregion_inds"])
    srk = set(sr["event_id"].astype(str)) if "event_id" in sr.columns else set()
    print(f"  spliceregion_inds covers {len(events & srk)}/{len(events)} events"
          f" ({100*len(events & srk)/max(len(events),1):.1f}%)")
    ec = pd.read_csv(pf["events_coordinates"], sep=None, engine="python")
    eck = set(ec["event_id"].astype(str)) if "event_id" in ec.columns else set()
    print(f"  events_coordinates covers {len(events & eck)}/{len(events)} events"
          f" ({100*len(events & eck)/max(len(events),1):.1f}%)")

    section("encoded sequence coverage (this is the usual failure mode)")
    try:
        import torch
        seqd = torch.load(pf["enc_seq_file"], map_location="cpu", weights_only=False)
        ks = set(map(str, seqd.keys()))
        by_gene = len(genes & ks); by_event = len(events & ks)
        mode = "gene_id" if by_gene >= by_event else "event_id"
        tgt = genes if mode == "gene_id" else events
        print(f"  enc_seq keyed by {mode}: covers {len(tgt & ks)}/{len(tgt)}"
              f" ({100*len(tgt & ks)/max(len(tgt),1):.1f}%)")
        if len(tgt & ks) < len(tgt):
            print(f"    missing examples: {sorted(tgt - ks)[:5]}")
            ok = False
    except Exception as e:
        print(f"  SKIP enc_seq check: {type(e).__name__}: {str(e)[:120]}")

    if a.deep:
        section("structure pickle + expression scatter (full model only)")
        if a.structure_pkl:
            try:
                import pickle
                with open(a.structure_pkl, "rb") as f:
                    sd = pickle.load(f)
                inner = sd[list(sd.keys())[0]] if isinstance(sd, dict) else {}
                sk = set(map(str, (inner.keys() if isinstance(inner, dict) else sd.keys())))
                print(f"  structure pkl covers {len(events & sk)}/{len(events)} events")
            except Exception as e:
                print(f"  structure pkl unreadable: {type(e).__name__}: {str(e)[:100]}")
        if a.scatter_dir:
            have = {f.replace("scatter_coeffs_", "").replace(".pt", "")
                    for f in os.listdir(a.scatter_dir) if f.startswith("scatter_coeffs_")}
            print(f"  scatter dir covers {len(neurons & have)}/{len(neurons)} neurons")
            if neurons - have:
                print(f"    missing: {sorted(neurons - have)[:8]}")

    print("\n" + ("PREFLIGHT PASS -- safe to submit csn_seqonly on this config"
                  if ok else
                  "PREFLIGHT FAIL -- see FAIL/missing lines above"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
