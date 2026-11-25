"""Build gene_symbol -> ENSG (no version) from GENCODE GTF (protein_coding genes)."""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator


def _iter_gene_lines(gtf_path: str) -> Iterator[str]:
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if "\tgene\t" not in line:
                continue
            yield line


def parse_gencode_gene_symbols(
    gtf_path: str,
    prefer_types: tuple[str, ...] = (
        "protein_coding",
        "lncRNA",
        "miRNA",
        "snoRNA",
        "snRNA",
        "Mt_rRNA",
        "Mt_tRNA",
        "antisense",
        "sense_intronic",
        "sense_overlapping",
        "3prime_overlapping_ncRNA",
        "lincRNA",
        "TEC",
        "transcribed_unprocessed_pseudogene",
        "transcribed_processed_pseudogene",
        "unprocessed_pseudogene",
        "processed_pseudogene",
        "unitary_pseudogene",
        "polymorphic_pseudogene",
        "IG_V_pseudogene",
        "TR_V_pseudogene",
    ),
) -> dict[str, str]:
    """
    Map HGNC symbol -> ENSG ID (no version suffix).
    Prefers protein_coding, then lncRNA, then other biotypes (so RNU/snRNA genes resolve).
    Any gene_type not listed gets rank 50 (after prefer_types).
    """
    gene_id_re = re.compile(r'gene_id "([^"]+)"')
    name_re = re.compile(r'gene_name "([^"]+)"')
    type_re = re.compile(r'gene_type "([^"]+)"')

    ranked = {t: i for i, t in enumerate(prefer_types)}
    default_rank = len(prefer_types) + 10

    candidates: dict[str, list[tuple[int, str]]] = {}

    for line in _iter_gene_lines(gtf_path):
        gid_m = gene_id_re.search(line)
        name_m = name_re.search(line)
        type_m = type_re.search(line)
        if not (gid_m and name_m and type_m):
            continue
        ensg = gid_m.group(1).split(".")[0]
        symbol = name_m.group(1)
        gtype = type_m.group(1)
        rank = ranked.get(gtype, default_rank)
        candidates.setdefault(symbol, []).append((rank, ensg))

    symbol_to_ensg: dict[str, str] = {}
    for symbol, lst in candidates.items():
        lst.sort(key=lambda x: x[0])
        symbol_to_ensg[symbol] = lst[0][1]

    return symbol_to_ensg
