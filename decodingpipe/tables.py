from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import AA_TO_CODONS, GENETIC_CODE_RNA, SENSE_CODONS
from .fasta import CodonCountingOptions, codon_table_for_locus_tags, compute_per_gene_rcu_matrix
from .utils import norm_str


def build_tables_from_fasta(
    *,
    clusters: Dict[str, List[str]],
    allgenes: List[str],
    codon_map: pd.DataFrame,
    seq_by_lt: Dict[str, str],
    opts: CodonCountingOptions,
    beta: float,
    exclude_aa: set[str],
    allow_aa_composition_shift: bool,
    aa_totals_use_cluster_gene_counts: bool,
    pseudocount_per_gene: int,
):
    """Build per-cluster codon tables for both metrics: pooled-count RCU and devZ-reconstructed RCU_devZ."""

    # ---- pooled RCU (counts) ----
    codon_tables_rcu: Dict[str, pd.DataFrame] = {}
    qc_rows_rcu = []

    df_all, sum_all = codon_table_for_locus_tags(allgenes, seq_by_lt, opts)
    df_all = df_all.merge(codon_map, on=["Codon", "AA3"], how="left")
    df_all["Mean_RCU_devZ"] = 0.0
    df_all["Mean_RCU"] = np.nan
    codon_tables_rcu["All genes"] = df_all
    qc_rows_rcu.append({"Cluster": "All genes", **sum_all})

    for cl_name, tags in clusters.items():
        df_cl, sum_cl = codon_table_for_locus_tags(tags, seq_by_lt, opts)
        df_cl = df_cl.merge(codon_map, on=["Codon", "AA3"], how="left")
        df_cl["Mean_RCU_devZ"] = np.nan
        df_cl["Mean_RCU"] = np.nan
        codon_tables_rcu[str(cl_name)] = df_cl
        qc_rows_rcu.append({"Cluster": str(cl_name), **sum_cl})

    qc_rcu = pd.DataFrame(qc_rows_rcu)

    # ---- per-gene RCU -> devZ -> pseudo-count reconstruction ----
    gene_rcu_df, _missing = compute_per_gene_rcu_matrix(allgenes, seq_by_lt, opts, exclude_aa=exclude_aa)

    rcu_mat = gene_rcu_df.values
    mean = np.nanmean(rcu_mat, axis=0)
    std = np.nanstd(rcu_mat, axis=0, ddof=1)
    std_safe = std.copy()
    std_safe[~np.isfinite(std_safe)] = 0.0

    devz = np.full_like(rcu_mat, np.nan, dtype=float)
    for j in range(rcu_mat.shape[1]):
        col = rcu_mat[:, j]
        m = mean[j]
        s = std_safe[j]
        mask = np.isfinite(col)
        if not np.any(mask):
            continue
        devz[mask, j] = 0.0 if s == 0.0 else (col[mask] - m) / s

    gene_devz_df = pd.DataFrame(devz, index=gene_rcu_df.index, columns=gene_rcu_df.columns)

    # baseline p0 from observed all-genes counts
    p0_by_aa, aa_counts_genome = {}, {}
    for aa3, cods in AA_TO_CODONS.items():
        if aa3 in exclude_aa:
            continue
        sub = df_all[df_all["AA3"] == aa3]
        tot = float(sub["Count"].sum())
        aa_counts_genome[aa3] = tot
        if tot <= 0:
            continue
        p0_by_aa[aa3] = {
            c: float(sub.loc[sub["Codon"] == c, "Count"].values[0]) / tot
            for c in cods if (sub["Codon"] == c).any()
        }

    def cluster_n_genes_with_aa(gene_subset_df: pd.DataFrame) -> dict:
        out = {}
        for aa3 in sorted(set(GENETIC_CODE_RNA[c] for c in SENSE_CODONS)):
            if aa3 in exclude_aa or aa3 == "Stop":
                continue
            cods = AA_TO_CODONS.get(aa3, [])
            if not cods:
                continue
            sub = gene_subset_df[cods]
            out[aa3] = int(sub.notna().any(axis=1).sum())
        return out

    def build_devz_table_for_cluster(cluster_name: str, gene_list: List[str]):
        genes_present = [g for g in gene_list if g in gene_devz_df.index]
        sub = gene_devz_df.loc[genes_present] if genes_present else gene_devz_df.iloc[0:0]
        zbar = sub.mean(axis=0, skipna=True).to_dict() if len(sub) else {c: 0.0 for c in SENSE_CODONS}
        n_with_aa = cluster_n_genes_with_aa(gene_rcu_df.loc[genes_present]) if genes_present else {}

        def total_aa(aa3: str) -> float:
            if allow_aa_composition_shift and aa_totals_use_cluster_gene_counts:
                return float(n_with_aa.get(aa3, 0)) * float(pseudocount_per_gene)
            return float(aa_counts_genome.get(aa3, 0.0))

        eps = 1e-12
        p_cluster = {}
        for aa3, cods in AA_TO_CODONS.items():
            if aa3 in exclude_aa:
                continue
            tot = total_aa(aa3)
            if tot <= 0:
                for c in cods:
                    p_cluster[c] = 0.0
                continue
            ws = []
            for c in cods:
                p0 = float(p0_by_aa.get(aa3, {}).get(c, 1.0 / max(1, len(cods))))
                zc = float(zbar.get(c, 0.0)) if np.isfinite(zbar.get(c, np.nan)) else 0.0
                ws.append((p0 + eps) * float(np.exp(beta * zc)))
            wsum = float(np.sum(ws))
            if wsum <= 0:
                for c in cods:
                    p_cluster[c] = 1.0 / len(cods)
            else:
                for c, w in zip(cods, ws):
                    p_cluster[c] = float(w / wsum)

        rows = []
        for codon in SENSE_CODONS:
            aa3 = GENETIC_CODE_RNA[codon]
            if aa3 in exclude_aa:
                continue
            rows.append({"Codon": codon, "AA3": aa3, "Mean_RCU_devZ": float(zbar.get(codon, 0.0))})
        df = pd.DataFrame(rows)

        counts = []
        for _, r in df.iterrows():
            aa3 = r["AA3"]
            counts.append(total_aa(aa3) * float(p_cluster.get(r["Codon"], 0.0)))

        df["Count"] = np.round(np.array(counts, dtype=float)).astype(int)
        total_c = int(df["Count"].sum())
        df["Freq"] = df["Count"] / total_c if total_c > 0 else np.nan

        df["nSyn"] = df["AA3"].map(lambda aa: len(AA_TO_CODONS.get(aa, []))).astype(int)
        aa_totals = df.groupby("AA3")["Count"].transform("sum")
        expected = aa_totals / df["nSyn"].replace(0, np.nan)
        df["RSCU"] = df["Count"] / expected
        df.loc[df["nSyn"] <= 1, "RSCU"] = 1.0

        aa_tot = df.groupby("AA3")["Count"].transform("sum").replace(0, np.nan)
        df["RCU"] = df["Count"] / aa_tot

        df["Mean_RCU"] = np.nan
        df = df.merge(codon_map, on=["Codon", "AA3"], how="left")

        qc = {"Cluster": cluster_name, "n_genes_requested": len(gene_list),
              "n_genes_found_in_fasta": len(genes_present),
              "n_genes_missing_in_fasta": len(gene_list) - len(genes_present),
              "total_codons_kept": total_c}
        return df.sort_values(["AA3", "Codon"]).reset_index(drop=True), qc

    codon_tables_devz: Dict[str, pd.DataFrame] = {}
    qc_rows_devz = []

    df_all_devz = df_all.copy()
    df_all_devz["Mean_RCU_devZ"] = 0.0
    df_all_devz["Mean_RCU"] = np.nan
    codon_tables_devz["All genes"] = df_all_devz
    qc_rows_devz.append({"Cluster": "All genes",
                         "n_genes_requested": len(allgenes),
                         "n_genes_found_in_fasta": sum_all["n_genes_found_in_fasta"],
                         "n_genes_missing_in_fasta": sum_all["n_genes_missing_in_fasta"],
                         "total_codons_kept": int(df_all_devz["Count"].sum())})

    for cl_name, tags in clusters.items():
        df_cl_devz, qc = build_devz_table_for_cluster(str(cl_name), tags)
        codon_tables_devz[str(cl_name)] = df_cl_devz
        qc_rows_devz.append(qc)

    qc_devz = pd.DataFrame(qc_rows_devz)

    return {"rcu": codon_tables_rcu, "rcu_devz": codon_tables_devz}, {"rcu": qc_rcu, "rcu_devz": qc_devz}
