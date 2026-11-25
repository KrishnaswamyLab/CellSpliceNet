#!/usr/bin/env python3
"""Per-exon sequence features used by the feature baselines (GBM) and the exon-feature arms.

Inputs:  <data_root>/human/events.csv, raw/ascot/gtex_psi.csv.gz, raw/hg38/GRCh38.p14.genome.fa
Output:  <data_root>/human/processed/event_features.csv
         event_id, chrom, exon_len, up_intron, dn_intron, log_exon, log_up, log_dn, len_mod3,
         gc_exon, gc_up, gc_dn (exon / 100 nt flanks), maxent3, maxent5 (MaxEntScan acceptor / donor),
         ag_up (AG count in the last 30 nt of the upstream intron)
Needs maxentpy (pip install git+https://github.com/kepbod/maxentpy).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import organism_paths  # noqa: E402
from processing.human.genome import Genome, parse_interval  # noqa: E402

FLANK = 300
COMP = str.maketrans("ACGTN", "TGCAN")


def gc(s):
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def main():
    from maxentpy import maxent
    raw, processed, out = organism_paths("human")
    events = pd.read_csv(processed.parent / "events.csv", usecols=["event_id"]).event_id.unique()
    asc = pd.read_csv(raw / "ascot/gtex_psi.csv.gz", encoding="utf-8-sig",
                      usecols=["exon_id", "exon_strand", "exon_location", "exon_boundary"]).set_index("exon_id")
    asc = asc.reindex(events)
    g = Genome(raw / "hg38/GRCh38.p14.genome.fa")
    m3, m5 = maxent.load_matrix3(), maxent.load_matrix5()
    rows = []
    for e, r in asc.iterrows():
        if not isinstance(r.exon_location, str):
            continue
        ch, a, b = parse_interval(r.exon_location)
        _, ba, bb = parse_interval(r.exon_boundary)
        up_len, dn_len = (a - ba, bb - b) if r.exon_strand == "+" else (bb - b, a - ba)
        s = g.fetch(ch, a - FLANK - 1, b + FLANK)
        if r.exon_strand == "-":
            s = s.translate(COMP)[::-1]
        L = b - a + 1
        ex, up, dn = s[FLANK:FLANK + L], s[FLANK - 100:FLANK], s[FLANK + L:FLANK + L + 100]
        acc, don = s[FLANK - 20:FLANK + 3], s[FLANK + L - 3:FLANK + L + 6]
        try:
            s3 = maxent.score3(acc, matrix=m3) if len(acc) == 23 and "N" not in acc else np.nan
        except Exception:
            s3 = np.nan
        try:
            s5 = maxent.score5(don, matrix=m5) if len(don) == 9 and "N" not in don else np.nan
        except Exception:
            s5 = np.nan
        rows.append(dict(event_id=e, chrom=ch, exon_len=L, up_intron=up_len, dn_intron=dn_len, log_exon=np.log10(L),
                         log_up=np.log10(max(up_len, 1)), log_dn=np.log10(max(dn_len, 1)), len_mod3=L % 3,
                         gc_exon=gc(ex), gc_up=gc(up), gc_dn=gc(dn), maxent3=s3, maxent5=s5,
                         ag_up=up[-30:].count("AG")))
    f = pd.DataFrame(rows)
    f.to_csv(out / "event_features.csv", index=False)
    print(f"event_features.csv: {len(f):,} events -> {out}", flush=True)


if __name__ == "__main__":
    main()
