#!/usr/bin/env python3
"""Z-scored copy of the tissue scatter coefficients, standardized with training-tissue statistics only.

Raw tissue scatter inputs are nearly identical across tissues (cosine >= 0.97); this removes the shared
component. z = (x - mean) / (sd + 1e-6) over the training tissues of splits/specs/human/cell.json,
0 where sd <= 1e-6, clipped to [-5, 5].

Input:   <data_root>/human/processed/expression/scatter_493/scatter_coeffs_<tissue>.pt
Output:  <data_root>/human/processed/expression/scatter_493_z_cell/scatter_coeffs_<tissue>.pt
         (use it with make_splits.py --expression scatter_493_z_cell)
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from processing.common import organism_paths  # noqa: E402
from processing.human.event_variability import training_tissues  # noqa: E402


def main():
    raw, processed, out = organism_paths("human")
    src = processed / "expression/scatter_493"
    dst = out / "expression/scatter_493_z_cell"
    dst.mkdir(parents=True, exist_ok=True)
    tissues = sorted(pd.read_csv(processed.parent / "events.csv", usecols=["neuron"]).neuron.unique())
    A = {t: torch.load(src / f"scatter_coeffs_{t}.pt", map_location="cpu").float() for t in tissues}
    M = torch.stack([A[t] for t in training_tissues(tissues)])
    mu, sd = M.mean(0), M.std(0)
    for t in tissues:
        z = torch.where(sd > 1e-6, (A[t] - mu) / (sd + 1e-6), torch.zeros_like(mu)).clamp(-5, 5)
        torch.save(z, dst / f"scatter_coeffs_{t}.pt")
    print(f"{len(tissues)} tissues -> {dst}", flush=True)


if __name__ == "__main__":
    main()
