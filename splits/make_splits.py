#!/usr/bin/env python3
"""Make a train/valid/test split from <data_root>/<organism>/events.csv.

    python splits/make_splits.py --organism human --split chrom
    python splits/make_splits.py --organism human --split gene --max_cells_per_event 8      # -> gene_k8
    python splits/make_splits.py --organism c_elegans --split cell --target mean

Writes <data_root>/<organism>/splits/<name>/{train,valid,test}.csv, split.json and data_config.ini
(pass the latter to --config_fname). Split definitions are in splits/specs/<organism>/<split>.json:

  gene_disjoint   genes shuffled with a fixed seed and cut into train/valid/test fractions
  chrom           held-out chromosomes; exons similar to held-out ones (megablast) removed from train/valid
  cell            held-out tissues / neurons on top of a gene split; test marks gene_seen
  gene_x_cell     held-out chromosomes AND tissues; test marks gene_seen / tissue_seen
  frozen          a saved (event, neuron) -> split assignment (historical worm splits, made unseeded)

C. elegans targets (--target): mean = PSI averaged over replicates with reads, nb_reads averaged over
all replicates (the published per-neuron data); sample1 = the one replicate per (event, neuron) saved
with the split (reproduces the 2026 held-out splits); paper_table = the published per-neuron table,
used by the original row split.
"""
import argparse
import configparser
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from processing.common import default_data_root  # noqa: E402

SPLITS = ("train", "valid", "test")
COLS = ["gene_id", "event_id", "PSI", "neuron", "nb_reads", "p_seq_len", "sj_seq_len",
        "exon_start", "exon_end", "intron_start", "intron_end", "gene_length"]


# ----------------------------------------------------------------------------- targets / tables
def worm_table(root, target, frozen=None):
    """Per-(event, neuron) C. elegans table for the requested target."""
    org = root / "c_elegans"
    coords = pd.read_csv(org / "processed/events_coordinates.tsv", sep="\t").set_index("event_id")
    if target == "paper_table":
        t = pd.read_csv(next((org / "raw").glob("2024-12-04-*_main_data.csv")))
        for c in ("exon_start", "exon_end", "intron_start", "intron_end", "gene_length"):
            t[c] = t.event_id.map(coords[c])
        return t
    ev = pd.read_csv(org / "events.csv")
    if target == "sample1":
        if frozen is None or "neuron_replicate" not in frozen:
            raise SystemExit("--target sample1 needs a frozen split that saved the sampled replicate")
        keep = frozen[["event_id", "neuron", "neuron_replicate"]].rename(columns={"neuron_replicate": "replicate"})
        t = ev.merge(keep, on=["event_id", "neuron", "replicate"]).rename(columns={"replicate": "neuron_replicate"})
        return t[t.PSI.notna()]
    g = ev.groupby(["event_id", "neuron"], sort=False)
    t = g.agg(PSI=("PSI", "mean"), nb_reads=("nb_reads", "mean"), gene_id=("gene_id", "first"),
              p_seq_len=("p_seq_len", "first"), sj_seq_len=("sj_seq_len", "first"),
              exon_start=("exon_start", "first"), exon_end=("exon_end", "first"),
              intron_start=("intron_start", "first"), intron_end=("intron_end", "first"),
              gene_length=("gene_length", "first")).reset_index()
    return t[t.PSI.notna()]


def human_table(root):
    return pd.read_csv(root / "human/events.csv")


# ----------------------------------------------------------------------------- split kinds
def gene_disjoint(t, spec):
    rng = np.random.default_rng(spec["seed"])
    genes = np.array(sorted(set(t.gene_id)))
    rng.shuffle(genes)
    n_tr, n_va = int(len(genes) * spec["fractions"][0]), int(len(genes) * spec["fractions"][1])
    part = {**{g: "train" for g in genes[:n_tr]}, **{g: "valid" for g in genes[n_tr:n_tr + n_va]},
            **{g: "test" for g in genes[n_tr + n_va:]}}
    return {k: t[t.gene_id.map(part) == k] for k in SPLITS}


def chrom(t, spec):
    test, valid = set(spec["test_chroms"]), set(spec["valid_chroms"])
    part = t.chrom.map(lambda c: "test" if c in test else ("valid" if c in valid else "train"))
    drop = pd.read_csv(HERE / spec["homology_drop"])
    out = {}
    for k in SPLITS:
        gone = set(drop.event_id[drop.dropped_from == k])
        out[k] = t[(part == k) & ~t.event_id.isin(gone)]
    return out


def cell(t, spec, specs_dir):
    base = make(t, load_spec(specs_dir, spec["base"]), specs_dir)
    valid_c, test_c = set(spec["valid_cells"]), set(spec["test_cells"])
    train_c = set(t.neuron) - valid_c - test_c
    seen = set(base["train"].gene_id)
    out = {"train": base["train"][base["train"].neuron.isin(train_c)],
           "valid": base["train"][base["train"].neuron.isin(valid_c)],
           "test": t[t.neuron.isin(test_c)].copy()}
    out["test"]["gene_seen"] = out["test"].gene_id.isin(seen).astype(int)
    return out


def gene_x_cell(t, spec, specs_dir):
    genes = make(t, load_spec(specs_dir, spec["genes"]), specs_dir)
    cells = load_spec(specs_dir, spec["cells"])
    valid_c, test_c = set(cells["valid_cells"]), set(cells["test_cells"])
    train_c = set(t.neuron) - valid_c - test_c
    tr = genes["train"][genes["train"].neuron.isin(train_c)]
    seen = set(tr.event_id)
    va = genes["valid"][genes["valid"].neuron.isin(train_c | valid_c)].copy()
    te = pd.concat([genes["test"][genes["test"].neuron.isin(train_c | test_c)],
                    genes["train"][genes["train"].neuron.isin(test_c) & genes["train"].event_id.isin(seen)]])
    for x in (va, te):
        x["gene_seen"] = x.event_id.isin(seen).astype(int)
        x["tissue_seen"] = x.neuron.isin(train_c).astype(int)
    return {"train": tr, "valid": va, "test": te}


def frozen(t, spec, assignment):
    a = assignment
    if "neuron_replicate" not in t:          # replicate-averaged targets: the saved replicate does not apply
        a = a.drop(columns="neuron_replicate", errors="ignore")
    if spec.get("event_types"):
        a = a[a.event_id.str.split("_").str[0].isin(spec["event_types"])]
    m = a.merge(t, on=["event_id", "neuron"] + (["neuron_replicate"] if "neuron_replicate" in t else []), how="left")
    return {k: m[(m.split == k) & m.PSI.notna()].drop(columns="split") for k in SPLITS}, int(m.PSI.isna().sum())


def load_spec(specs_dir, name):
    return json.load(open(specs_dir / f"{name}.json"))


def make(t, spec, specs_dir):
    kind = spec["kind"]
    if kind == "gene_disjoint":
        return gene_disjoint(t, spec)
    if kind == "chrom":
        return chrom(t, spec)
    if kind == "cell":
        return cell(t, spec, specs_dir)
    if kind == "gene_x_cell":
        return gene_x_cell(t, spec, specs_dir)
    raise ValueError(kind)


def subsample(train, k):
    """At most k cells (tissues / neurons) per event in the training set (random_state 0 per event)."""
    return train.groupby("event_id", group_keys=False).apply(lambda d: d.sample(min(k, len(d)), random_state=0))


# ----------------------------------------------------------------------------- main
DEFAULT_EXPRESSION = {"c_elegans": "scatter_243", "human": "scatter_493"}
EVENTS_COORDS = {"c_elegans": "events_coordinates.tsv", "human": "events_coordinates.csv"}


def write_config(out, organism, expression):
    c = configparser.ConfigParser()
    p = f"$DATASET/{organism}/processed"
    c["files"] = {"enc_seq_file": f"{p}/encoded_sequences.pth", "enc_sj_file": f"{p}/enc_sj_data.pth",
                  "spliceregion_inds": f"{p}/spliceregion_inds.csv",
                  "events_coordinates": f"{p}/{EVENTS_COORDS[organism]}",
                  "structure_data_root": f"{p}/structure/structure_scattering_dict.pkl",
                  "scatter_coeffs_dir": f"{p}/expression/{expression}",
                  **{f"{k}_data_file": f"$CONFIG_DIR/{k}.csv" for k in SPLITS}}
    with open(out / "data_config.ini", "w") as f:
        c.write(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--organism", choices=["c_elegans", "human"], required=True)
    ap.add_argument("--split", required=True, help="spec name in splits/specs/<organism>/")
    ap.add_argument("--target", choices=["mean", "sample1", "paper_table"], default=None,
                    help="C. elegans target (default: the spec's 'target', else mean)")
    ap.add_argument("--max_cells_per_event", type=int, default=0, help="subsample the training set (name gets _k<K>)")
    ap.add_argument("--expression", default=None, help="folder under processed/expression (default: scatter_243 / scatter_493)")
    ap.add_argument("--name", default=None, help="output folder name (default: derived from split, target, K)")
    ap.add_argument("--data_root", default=str(default_data_root()))
    a = ap.parse_args()

    root = Path(a.data_root)
    specs_dir = HERE / "specs" / a.organism
    spec = load_spec(specs_dir, a.split)
    note = {}
    if a.organism == "c_elegans":
        target = a.target or spec.get("target", "mean")
        assignment = pd.read_csv(HERE / spec["assignment"]) if spec["kind"] == "frozen" else None
        t = worm_table(root, target, assignment)
        if spec["kind"] == "frozen":
            parts, note["assigned_rows_without_target"] = frozen(t, spec, assignment)
        else:
            parts = make(t, spec, specs_dir)
    else:
        target = None
        parts = make(human_table(root), spec, specs_dir)
    if a.max_cells_per_event:
        parts["train"] = subsample(parts["train"], a.max_cells_per_event)

    name = a.name or (a.split + (f"_{target}" if target and target != spec.get("target", "mean") else "")
                      + (f"_k{a.max_cells_per_event}" if a.max_cells_per_event else ""))
    out = root / a.organism / "splits" / name
    out.mkdir(parents=True, exist_ok=True)
    info = {"organism": a.organism, "split": a.split, "spec": spec, "target": target,
            "max_cells_per_event": a.max_cells_per_event, **note, "files": {}}
    for k in SPLITS:
        d = parts[k]
        cols = [c for c in COLS + ["neuron_replicate", "chrom", "gene_seen", "tissue_seen"] if c in d.columns]
        d[cols].to_csv(out / f"{k}.csv", index=False)
        info["files"][k] = {"rows": len(d), "events": int(d.event_id.nunique()), "genes": int(d.gene_id.nunique()),
                            "cells": int(d.neuron.nunique()),
                            "md5": hashlib.md5((out / f"{k}.csv").read_bytes()).hexdigest()}
    write_config(out, a.organism, a.expression or DEFAULT_EXPRESSION[a.organism])
    json.dump(info, open(out / "split.json", "w"), indent=1)
    print(f"{a.organism}/splits/{name}: " + ", ".join(f"{k} {v['rows']:,} rows / {v['events']:,} events"
                                                   for k, v in info["files"].items()), flush=True)


if __name__ == "__main__":
    main()
