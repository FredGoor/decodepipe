from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .constants import AA_TO_CODONS, GENETIC_CODE_RNA, SENSE_CODONS, STOP_CODONS_DNA
from .utils import norm_str


_LT_PATTERNS = [
    re.compile(r"\[locus_tag=([^\]]+)\]"),
    re.compile(r"\blocus_tag\s*=\s*([A-Za-z0-9_.:\-]+)"),
    re.compile(r"\blocus_tag:([A-Za-z0-9_.:\-]+)"),
]


@dataclass(frozen=True)
class CodonCountingOptions:
    drop_terminal_stop: bool = True
    ignore_internal_stops: bool = True
    include_ambiguous_codons: bool = False


def extract_locus_tag_from_header(header: str) -> str:
    h = norm_str(header)
    if not h:
        return ""
    for pat in _LT_PATTERNS:
        m = pat.search(h)
        if m:
            return norm_str(m.group(1)).strip().strip(']"\'')
    if h.startswith(">"):
        token = h[1:].split()[0].strip()
        if token and ("|" not in token) and re.match(r"^[A-Za-z0-9_.:\-]+$", token):
            return token
    return ""


def read_fasta_cds(fasta_path: Path) -> Dict[str, str]:
    """Read CDS FASTA into a dict {locus_tag: DNA_sequence} (DNA alphabet, uppercase)."""
    seq_by_lt: Dict[str, str] = {}
    header = None
    seq_chunks: List[str] = []
    n_records = n_with_lt = n_without_lt = 0

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    n_records += 1
                    lt = extract_locus_tag_from_header(header)
                    if lt:
                        n_with_lt += 1
                        seq_by_lt[lt] = "".join(seq_chunks).upper().replace("U", "T")
                    else:
                        n_without_lt += 1
                header, seq_chunks = line, []
            else:
                seq_chunks.append(line)

        if header is not None:
            n_records += 1
            lt = extract_locus_tag_from_header(header)
            if lt:
                n_with_lt += 1
                seq_by_lt[lt] = "".join(seq_chunks).upper().replace("U", "T")
            else:
                n_without_lt += 1

    return seq_by_lt


def count_codons_in_seq(dna_seq: str, opts: CodonCountingOptions) -> Dict[str, int]:
    """Count sense codons (RNA alphabet) in a DNA sequence."""
    s = dna_seq.upper().replace("U", "T")
    if len(s) < 3:
        return {}
    trim = len(s) % 3
    if trim:
        s = s[:-trim]
    codons = [s[i:i + 3] for i in range(0, len(s), 3)]
    if opts.drop_terminal_stop and codons and codons[-1] in STOP_CODONS_DNA:
        codons = codons[:-1]

    counts: Dict[str, int] = {}
    for c in codons:
        if (not opts.include_ambiguous_codons) and re.search(r"[^ACGT]", c):
            continue
        if opts.ignore_internal_stops and c in STOP_CODONS_DNA:
            continue
        c_rna = c.replace("T", "U")
        aa = GENETIC_CODE_RNA.get(c_rna)
        if aa and aa != "Stop":
            counts[c_rna] = counts.get(c_rna, 0) + 1
    return counts


def codon_table_for_locus_tags(
    locus_tags: List[str],
    seq_by_lt: Dict[str, str],
    opts: CodonCountingOptions,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Pooled codon table for a set of locus tags."""
    total_counts = {c: 0 for c in SENSE_CODONS}
    n_missing = 0
    for lt in locus_tags:
        seq = seq_by_lt.get(lt)
        if seq is None:
            n_missing += 1
            continue
        counts = count_codons_in_seq(seq, opts)
        for codon, n in counts.items():
            if codon in total_counts:
                total_counts[codon] += int(n)

    total = int(sum(total_counts.values()))
    df = pd.DataFrame({"Codon": SENSE_CODONS})
    df["AA3"] = df["Codon"].map(GENETIC_CODE_RNA)
    df["Count"] = df["Codon"].map(total_counts).fillna(0).astype(int)
    df["Freq"] = df["Count"] / total if total > 0 else np.nan

    df["nSyn"] = df["AA3"].map(lambda aa: len(AA_TO_CODONS.get(aa, [])))
    aa_totals = df.groupby("AA3")["Count"].transform("sum")
    expected = aa_totals / df["nSyn"].replace(0, np.nan)
    df["RSCU"] = df["Count"] / expected
    df.loc[df["nSyn"] <= 1, "RSCU"] = 1.0

    aa_tot = df.groupby("AA3")["Count"].transform("sum").replace(0, np.nan)
    df["RCU"] = df["Count"] / aa_tot

    summary = {
        "n_genes_requested": int(len(locus_tags)),
        "n_genes_found_in_fasta": int(len(locus_tags) - n_missing),
        "n_genes_missing_in_fasta": int(n_missing),
        "total_codons_kept": int(total),
    }
    return df.sort_values(["AA3", "Codon"]).reset_index(drop=True), summary


def compute_per_gene_rcu_matrix(
    gene_list: List[str],
    seq_by_lt: Dict[str, str],
    opts: CodonCountingOptions,
    exclude_aa: set[str],
) -> Tuple[pd.DataFrame, int]:
    """Per-gene RCU matrix: rows=genes, cols=codons (RNA), values within-AA fractions."""
    codons = SENSE_CODONS
    codon_idx = {c: i for i, c in enumerate(codons)}
    mat = np.full((len(gene_list), len(codons)), np.nan, dtype=float)
    aa_codons = {aa: AA_TO_CODONS.get(aa, []) for aa in AA_TO_CODONS.keys()}

    missing = 0
    for gi, lt in enumerate(gene_list):
        seq = seq_by_lt.get(lt)
        if seq is None:
            missing += 1
            continue
        counts = count_codons_in_seq(seq, opts)
        for aa, cod_list in aa_codons.items():
            if aa in exclude_aa:
                continue
            tot = sum(int(counts.get(c, 0)) for c in cod_list)
            if tot <= 0:
                continue
            for c in cod_list:
                mat[gi, codon_idx[c]] = float(counts.get(c, 0)) / float(tot)

    return pd.DataFrame(mat, index=gene_list, columns=codons), missing
