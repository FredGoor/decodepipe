from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .constants import AA_TO_CODONS, GENETIC_CODE_RNA, SENSE_CODONS
from .fasta import CodonCountingOptions, codon_table_for_locus_tags
from .utils import norm_str


def dna_to_rna_codon(x: str) -> str:
    s = norm_str(x).upper()
    if not s:
        return ""
    s = s.replace("U", "T")
    s = re.sub(r"[^ACGT]", "", s)
    if len(s) != 3:
        return ""
    return s.replace("T", "U")


def load_precomputed_codonbias_workbook(
    xlsx_path: Path,
    *,
    baseline_sheet: str,
    exclude_sheets: list[str],
    exclude_aa: set[str],
) -> Tuple[str, Dict[str, dict]]:
    """Read a per-gene codon-bias workbook.

    Expected sheet layout:
      - row 1: AA labels (from col 2)
      - row 2: codons (DNA; from col 2)
      - rows 4..: gene rows (col 1: gene id), values from col 2

    Returns:
      baseline_sheet_name, per_sheet dict
    """
    xl = pd.ExcelFile(xlsx_path)
    sheet_names = xl.sheet_names
    baseline = baseline_sheet.strip() if norm_str(baseline_sheet) else sheet_names[0]
    if baseline not in sheet_names:
        baseline = sheet_names[0]

    per_sheet: Dict[str, dict] = {}
    for sh in sheet_names:
        if sh != baseline and sh in (exclude_sheets or []):
            continue

        raw = pd.read_excel(xlsx_path, sheet_name=sh, header=None)
        if raw.shape[0] < 4 or raw.shape[1] < 2:
            continue

        aa_row = raw.iloc[0, 1:]
        codon_row = raw.iloc[1, 1:]

        codons_rna = []
        aa3_by_codon = {}
        col_idx_by_codon = {}

        for j, (aa3, cod_dna) in enumerate(zip(aa_row.values, codon_row.values), start=1):
            cod = dna_to_rna_codon(cod_dna)
            if not cod:
                continue
            aa3s = norm_str(aa3) or GENETIC_CODE_RNA.get(cod, "")
            if not aa3s or aa3s == "Stop":
                continue
            if aa3s in exclude_aa:
                continue
            if cod in col_idx_by_codon:
                continue
            col_idx_by_codon[cod] = j
            aa3_by_codon[cod] = aa3s
            codons_rna.append(cod)

        gene_col = raw.iloc[3:, 0].astype(str).str.strip()
        genes = [g for g in gene_col.tolist() if g and g.lower() != "nan"]

        block = raw.iloc[3:, [col_idx_by_codon[c] for c in codons_rna]].copy()
        block.columns = codons_rna
        block = block.apply(pd.to_numeric, errors="coerce")

        mean_bias = block.mean(axis=0, skipna=True).to_dict()
        median_bias = block.median(axis=0, skipna=True).to_dict()

        n_genes_with_aa = {}
        for aa3 in sorted(set(aa3_by_codon.values())):
            cods = [c for c in codons_rna if aa3_by_codon.get(c) == aa3]
            if not cods:
                continue
            n_genes_with_aa[aa3] = int(block[cods].notna().any(axis=1).sum())

        per_sheet[sh] = {
            "genes": genes,
            "mean_bias": mean_bias,
            "median_bias": median_bias,
            "n_genes_with_aa": n_genes_with_aa,
            "aa3_by_codon": aa3_by_codon,
            "codons_in_sheet": codons_rna,
        }

    return baseline, per_sheet


def baseline_p0_from_fasta(
    allgenes: list[str],
    seq_by_lt: dict,
    opts: CodonCountingOptions,
    exclude_aa: set[str],
):
    df_all, sum_all = codon_table_for_locus_tags(allgenes, seq_by_lt, opts)
    df_all = df_all.loc[~df_all["AA3"].isin(exclude_aa)].copy()

    p0_by_aa, aa_counts = {}, {}
    for aa3, cods in AA_TO_CODONS.items():
        if aa3 in exclude_aa:
            continue
        sub = df_all[df_all["AA3"] == aa3]
        tot = float(sub["Count"].sum())
        aa_counts[aa3] = tot
        if tot <= 0:
            continue
        p0_by_aa[aa3] = {
            c: float(sub.loc[sub["Codon"] == c, "Count"].values[0]) / tot
            for c in cods if (sub["Codon"] == c).any()
        }
    return p0_by_aa, aa_counts, df_all, sum_all


def build_tables_from_bias_precomputed(
    *,
    per_sheet: dict,
    baseline_sheet: str,
    codon_map: pd.DataFrame,
    seq_by_lt: dict,
    opts: CodonCountingOptions,
    metric_kind: str,  # "rcu" or "rcu_devz"
    z_agg: str,
    beta: float,
    exclude_aa: set[str],
    allow_aa_composition_shift: bool,
    aa_totals_use_cluster_gene_counts: bool,
    pseudocount_per_gene: int,
):
    """Build per-cluster codon tables from a precomputed per-gene workbook."""
    allgenes = per_sheet[baseline_sheet]["genes"]
    p0_by_aa, aa_counts_genome, df_all_counts, sum_all = baseline_p0_from_fasta(
        allgenes, seq_by_lt, opts, exclude_aa
    )

    def total_aa_for(aa3: str, info: dict) -> float:
        if allow_aa_composition_shift and aa_totals_use_cluster_gene_counts:
            n_aa = float(info["n_genes_with_aa"].get(aa3, 0))
            return n_aa * float(pseudocount_per_gene)
        return float(aa_counts_genome.get(aa3, 0.0))

    def build_one(sheet_name: str):
        info = per_sheet[sheet_name]
        bias = info["mean_bias"] if z_agg.lower() == "mean" else info["median_bias"]

        rows = []
        for codon in SENSE_CODONS:
            aa3 = GENETIC_CODE_RNA.get(codon, "")
            if not aa3 or aa3 == "Stop" or aa3 in exclude_aa:
                continue
            b = bias.get(codon, np.nan)
            rows.append({"Codon": codon, "AA3": aa3, "Bias": float(b) if np.isfinite(b) else np.nan})
        df = pd.DataFrame(rows)
        if df.empty:
            empty = pd.DataFrame(columns=["Codon","AA3","Count","Freq","RSCU","RCU","Mean_RCU_devZ","Mean_RCU"])
            qc = {"n_genes_requested": len(info["genes"]), "n_genes_found_in_fasta": np.nan,
                  "n_genes_missing_in_fasta": np.nan, "total_codons_kept": 0}
            return empty, qc

        eps = 1e-12
        p_cluster = {}
        for aa3, cods in AA_TO_CODONS.items():
            if aa3 in exclude_aa:
                continue
            cods = [c for c in cods if c in set(df["Codon"].values)]
            if not cods:
                continue

            if metric_kind == "rcu_devz":
                ws = []
                for c in cods:
                    p0 = float(p0_by_aa.get(aa3, {}).get(c, 1.0 / max(1, len(cods))))
                    zc = df.loc[df["Codon"] == c, "Bias"].values[0]
                    zc = float(zc) if np.isfinite(zc) else 0.0
                    ws.append((p0 + eps) * float(np.exp(beta * zc)))
                wsum = float(np.sum(ws))
                if wsum <= 0:
                    for c in cods:
                        p_cluster[c] = 1.0 / len(cods)
                else:
                    for c, w in zip(cods, ws):
                        p_cluster[c] = float(w / wsum)

            elif metric_kind == "rcu":
                vals = []
                for c in cods:
                    v = df.loc[df["Codon"] == c, "Bias"].values[0]
                    vals.append(float(v) if np.isfinite(v) else 0.0)
                s = float(np.sum(vals))
                if s <= 0:
                    for c in cods:
                        p_cluster[c] = float(p0_by_aa.get(aa3, {}).get(c, 1.0 / len(cods)))
                else:
                    for c, v in zip(cods, vals):
                        p_cluster[c] = float(v / s)
            else:
                raise ValueError("metric_kind must be 'rcu' or 'rcu_devz'")

        counts = []
        for _, r in df.iterrows():
            aa3 = r["AA3"]
            cod = r["Codon"]
            counts.append(total_aa_for(aa3, info) * float(p_cluster.get(cod, 0.0)))

        df["Count"] = np.round(np.array(counts, dtype=float)).astype(int)
        total = int(df["Count"].sum())
        df["Freq"] = df["Count"] / total if total > 0 else np.nan

        df["nSyn"] = df["AA3"].map(lambda aa: len(AA_TO_CODONS.get(aa, []))).astype(int)
        aa_totals = df.groupby("AA3")["Count"].transform("sum")
        expected = aa_totals / df["nSyn"].replace(0, np.nan)
        df["RSCU"] = df["Count"] / expected
        df.loc[df["nSyn"] <= 1, "RSCU"] = 1.0

        aa_tot = df.groupby("AA3")["Count"].transform("sum").replace(0, np.nan)
        df["RCU"] = df["Count"] / aa_tot

        if metric_kind == "rcu_devz":
            df["Mean_RCU_devZ"] = df["Bias"].fillna(0.0)
            df["Mean_RCU"] = np.nan
        else:
            df["Mean_RCU"] = df["Bias"]
            df["Mean_RCU_devZ"] = np.nan

        qc = {"n_genes_requested": len(info["genes"]), "n_genes_found_in_fasta": np.nan,
              "n_genes_missing_in_fasta": np.nan, "total_codons_kept": total}
        df = df.merge(codon_map, on=["Codon", "AA3"], how="left")
        return df.sort_values(["AA3","Codon"]).reset_index(drop=True), qc

    codon_tables = {}
    qc_rows = []

    df_all = df_all_counts.copy()
    df_all["Mean_RCU_devZ"] = 0.0
    df_all["Mean_RCU"] = np.nan
    df_all = df_all.merge(codon_map, on=["Codon","AA3"], how="left")
    codon_tables["All genes"] = df_all
    qc_rows.append({"Cluster": "All genes", **sum_all})

    for sh in per_sheet:
        if sh == baseline_sheet:
            continue
        df_cl, qc = build_one(sh)
        codon_tables[sh] = df_cl
        qc_rows.append({"Cluster": sh, **qc})

    return codon_tables, pd.DataFrame(qc_rows)
