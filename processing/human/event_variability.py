#!/usr/bin/env python3
"""How much each exon's PSI varies across tissues (used to stratify evaluation and to select exons).

Inputs:  <data_root>/human/events.csv; training tissues = all tissues minus the validation and test
         tissues of splits/specs/human/cell.json (the 34 training tissues of the cell / gene_x_cell splits)
Output:  <data_root>/human/processed/event_variability.csv
         event_id, n_tissues, mean, sd_all, range_all (over all tissues), sd_train_tissues
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from processing.common import organism_paths  # noqa: E402


def training_tissues(tissues):
    spec = json.load(open(REPO / "splits/specs/human/cell.json"))
    return sorted(set(tissues) - set(spec["valid_cells"]) - set(spec["test_cells"]))


def main():
    raw, processed, out = organism_paths("human")
    ev = pd.read_csv(processed.parent / "events.csv", usecols=["event_id", "neuron", "PSI"])
    W = ev.pivot_table(index="event_id", columns="neuron", values="PSI")
    tr = training_tissues(W.columns)
    v = pd.DataFrame({"n_tissues": W.notna().sum(1), "mean": W.mean(1), "sd_all": W.std(1),
                      "range_all": W.max(1) - W.min(1), "sd_train_tissues": W[tr].std(1)})
    v.to_csv(out / "event_variability.csv")
    print(f"event_variability.csv: {len(v):,} events, {len(tr)} training tissues -> {out}", flush=True)


if __name__ == "__main__":
    main()
