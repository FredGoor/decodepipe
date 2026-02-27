from __future__ import annotations

from typing import Dict, List, Set

# RNA genetic code (standard)
GENETIC_CODE_RNA: Dict[str, str] = {
    "UUU": "Phe", "UUC": "Phe", "UUA": "Leu", "UUG": "Leu",
    "UCU": "Ser", "UCC": "Ser", "UCA": "Ser", "UCG": "Ser",
    "UAU": "Tyr", "UAC": "Tyr", "UAA": "Stop", "UAG": "Stop",
    "UGU": "Cys", "UGC": "Cys", "UGA": "Stop", "UGG": "Trp",
    "CUU": "Leu", "CUC": "Leu", "CUA": "Leu", "CUG": "Leu",
    "CCU": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "CAU": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
    "CGU": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AUU": "Ile", "AUC": "Ile", "AUA": "Ile", "AUG": "Met",
    "ACU": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "AAU": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
    "AGU": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
    "GUU": "Val", "GUC": "Val", "GUA": "Val", "GUG": "Val",
    "GCU": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "GAU": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
    "GGU": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}

SENSE_CODONS: List[str] = [c for c, aa in GENETIC_CODE_RNA.items() if aa != "Stop"]

STOP_CODONS_DNA: Set[str] = {"TAA", "TAG", "TGA"}

AA_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in GENETIC_CODE_RNA.items():
    if aa != "Stop":
        AA_TO_CODONS.setdefault(aa, []).append(codon)
for aa in list(AA_TO_CODONS.keys()):
    AA_TO_CODONS[aa] = sorted(AA_TO_CODONS[aa])

# -----------------------------
# Defaults for plotting
# -----------------------------
# These defaults are chosen so that the pipeline produces informative figures
# out-of-the-box without requiring a configuration file.

# 2-codon amino acids commonly used for the WC vs Wobble within-AA heatmap.
DEFAULT_WC_WOBBLE_AA: List[str] = ["Asn", "Asp", "Cys", "His", "Phe", "Tyr"]

# Example tRNAs (by short name) used for the "tRNA box" heatmap.
DEFAULT_TRNA_BOX_TRNAS: List[str] = ["Arg1", "Asn1", "Asp1", "Cys1", "His1", "Ile1", "Phe1", "Tyr1"]

# Manual row orders for modification heatmaps.
DEFAULT_MOD34_MANUAL_ORDER: List[str] = ["k2C34", "Q34", "Cm34/Um34", "mnm5s2U34", "I34", "cmo5U34"]
DEFAULT_MOD37_MANUAL_ORDER: List[str] = ["i6A37", "t6A37", "m2A37", "m6A37", "m1G37"]
