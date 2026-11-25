"""hg38 access for the human builders (pysam if installed, else pyfaidx).

    g = Genome(raw / "hg38/GRCh38.p14.genome.fa")
    g.fetch("chr1", start0, end0)   # 0-based half-open, upper case, clipped to the chromosome
"""
from __future__ import annotations


class Genome:
    def __init__(self, fasta):
        try:
            import pysam
            self._fa = pysam.FastaFile(str(fasta))
            self._len = dict(zip(self._fa.references, self._fa.lengths))
            self._get = lambda ch, a, b: self._fa.fetch(ch, a, b).upper()
        except ImportError:
            from pyfaidx import Fasta
            self._fa = Fasta(str(fasta), as_raw=True, sequence_always_upper=True)
            self._len = {k: len(v) for k, v in self._fa.items()}
            self._get = lambda ch, a, b: self._fa[ch][a:b]

    def length(self, ch: str) -> int:
        return self._len[ch]

    def fetch(self, ch: str, start0: int, end0: int) -> str:
        a, b = max(start0, 0), min(end0, self._len[ch])
        return self._get(ch, a, b) if b > a else ""


def parse_interval(s):
    """'chr1:100-200' -> ('chr1', 100, 200) (ASCOT coordinates, 1-based inclusive)."""
    ch, xy = s.split(":")
    a, b = xy.split("-")
    return ch, int(a), int(b)
