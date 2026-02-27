from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .constants import (
    AA_TO_CODONS,
    GENETIC_CODE_RNA,
    SENSE_CODONS,
    DEFAULT_WC_WOBBLE_AA,
    DEFAULT_TRNA_BOX_TRNAS,
    DEFAULT_MOD34_MANUAL_ORDER,
    DEFAULT_MOD37_MANUAL_ORDER,
)
from .fasta import CodonCountingOptions, read_fasta_cds
from .precomputed import build_tables_from_bias_precomputed, load_precomputed_codonbias_workbook
from .tables import build_tables_from_fasta
from .plotting import (
    HeatmapStyle,
    PubFontsPt,
    PubGeometryCm,
    GenelevelOptions,
    pad_heatmap_columns,
    plot_pub_heatmap,
    build_codon_pairtype_map_for_trna_subset,
    compute_genelevel_wc_wobble_tables,
    plot_genelevel_bar,
    plot_genelevel_box,
)
from .utils import (
    canonical_mod,
    canonicalize_trna_name,
    group_bounds_from_groupnames,
    group_bounds_from_labels,
    norm_str,
    safe_sheetname,
    split_trna_field_names,
    wrap_label_no_break,
)

logger = logging.getLogger("decodingpipe")


# -----------------------------
# Minimal config wrapper
# -----------------------------


def _cfg_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class Cfg:
    """Lightweight dict-backed config.

    This project intentionally runs interactively (file pickers) and does not
    ship a TOML config file. We keep a tiny wrapper to avoid deeply nested
    dict access throughout the pipeline.
    """

    def __init__(self, raw: dict):
        self.raw = raw if isinstance(raw, dict) else {}

    def get(self, *keys, default=None):
        return _cfg_get(self.raw, *keys, default=default)

    def path(self, *keys, required: bool = False) -> Optional[Path]:
        s = norm_str(self.get(*keys, default=""))
        if not s:
            if required:
                raise ValueError(f"Missing required path: {'.'.join(keys)}")
            return None
        return Path(s).expanduser().resolve()

    @property
    def mode(self) -> str:
        return norm_str(self.get("inputs", "mode", default="fasta")).lower() or "fasta"

    @property
    def output_dir(self) -> Optional[Path]:
        s = norm_str(self.get("outputs", "output_dir", default=""))
        return Path(s).expanduser().resolve() if s else None

    @property
    def run_tag(self) -> str:
        return norm_str(self.get("outputs", "run_tag", default=""))

    @property
    def save_plots(self) -> bool:
        return bool(self.get("outputs", "save_plots", default=True))

    @property
    def show_plots(self) -> bool:
        return bool(self.get("outputs", "show_plots", default=True))

    @property
    def figure_format(self) -> str:
        return norm_str(self.get("outputs", "figure_format", default="png")).lower() or "png"

    @property
    def figure_dpi(self) -> int:
        try:
            return int(self.get("outputs", "figure_dpi", default=220))
        except Exception:
            return 220

    @property
    def baseline_label(self) -> str:
        return norm_str(self.get("labels", "baseline_label", default="Genome")) or "Genome"

    @property
    def colorbar_label(self) -> str:
        return norm_str(self.get("labels", "colorbar_label", default="Log2 fold change vs baseline"))


# -----------------------------
# Input loaders (kept here to reduce module sprawl)
# -----------------------------


_DECODING_TABLE_REQUIRED_COLUMNS = [
    "AA",
    "Codon 5'-3'",
    "Anticodon 5'-3'",
    "tRNAs",
    "Decoding",
    "Mod34",
    "Mod37",
    "Associated tRMEs",
]


def load_decoding_table(path: Path, sheet: str):
    """Parse the Codon-tRNAs decoding table and return annotation maps."""
    from .utils import pair_type, split_semicolon_list

    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in _DECODING_TABLE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Codon-tRNAs decoding table is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )

    df = df.copy()
    df["Codon"] = df["Codon 5'-3'"].astype(str).str.upper().str.replace("T", "U")
    df["AA3"] = df["AA"].astype(str).str.split("-").str[0]
    df["PairType"] = df["Decoding"].apply(pair_type)
    df["PairRank"] = df["PairType"].map({"WC": 0, "Wobble": 1}).fillna(9).astype(int)

    # Primary row per codon (prefer WC)
    df_sorted = df.sort_values(["Codon", "PairRank"])
    primary = df_sorted.groupby("Codon", as_index=False).first()

    # All decoders per codon
    tmp = df[["Codon", "tRNAs", "Anticodon 5'-3'", "Decoding", "PairType"]].drop_duplicates().copy()
    tmp["Decoder"] = (
        tmp["tRNAs"].astype(str)
        + "|"
        + tmp["Anticodon 5'-3'"].astype(str)
        + "|"
        + tmp["PairType"].astype(str)
    )
    all_decoders = tmp.groupby("Codon")["Decoder"].agg(lambda x: "; ".join(sorted(set(x))))

    def atoms_by_codon(colname: str) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for codon, g in df.groupby("Codon"):
            s: Set[str] = set()
            for v in g[colname].values:
                parts = split_semicolon_list(v) if isinstance(v, str) else [norm_str(v)]
                for p in parts:
                    p = norm_str(p)
                    if p and p.lower() not in {"none", "nan", "na", "n/a"}:
                        s.add(p)
            out[codon] = s
        return out

    mod34_by_codon = atoms_by_codon("Mod34")
    mod37_by_codon = atoms_by_codon("Mod37")

    trme_by_codon: Dict[str, Set[str]] = {}
    for codon, g in df.groupby("Codon"):
        s: Set[str] = set()
        for v in g["Associated tRMEs"].values:
            for p in split_semicolon_list(v):
                p = norm_str(p)
                if p:
                    s.add(p)
        trme_by_codon[codon] = s

    codon_map = primary[["Codon", "AA3", "tRNAs", "Anticodon 5'-3'", "Decoding", "PairType"]].copy()
    codon_map["AllDecoders"] = codon_map["Codon"].map(all_decoders).fillna("")
    codon_map["Mod34_atoms_str"] = codon_map["Codon"].map(
        lambda c: ";".join(sorted(mod34_by_codon.get(c, set()))) if mod34_by_codon.get(c) else "None"
    )
    codon_map["Mod37_atoms_str"] = codon_map["Codon"].map(
        lambda c: ";".join(sorted(mod37_by_codon.get(c, set()))) if mod37_by_codon.get(c) else "None"
    )
    codon_map["tRMEs_str"] = codon_map["Codon"].map(
        lambda c: ";".join(sorted(trme_by_codon.get(c, set()))) if trme_by_codon.get(c) else ""
    )

    codon_to_trna_set: Dict[str, Set[str]] = {c: set() for c in df["Codon"].unique()}
    codon_trna_to_pairtype: Dict[Tuple[str, str], str] = {}

    for _, r in df.iterrows():
        cod = norm_str(r["Codon"])
        pt = norm_str(r["PairType"])
        for trna in split_trna_field_names(r["tRNAs"]):
            if not trna:
                continue
            codon_to_trna_set.setdefault(cod, set()).add(trna)
            key = (cod, trna)
            if key not in codon_trna_to_pairtype:
                codon_trna_to_pairtype[key] = pt
            else:
                if codon_trna_to_pairtype[key] != "WC" and pt == "WC":
                    codon_trna_to_pairtype[key] = "WC"

    return codon_map, mod34_by_codon, mod37_by_codon, codon_to_trna_set, codon_trna_to_pairtype


def load_clusters(
    path: Path,
    *,
    sheet: str,
    genome_allgenes_column: str,
    exclude_cluster_columns: List[str] | None = None,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Load cluster gene lists from an Excel workbook."""
    exclude_cluster_columns = exclude_cluster_columns or []
    df = pd.read_excel(path, sheet_name=sheet)

    if genome_allgenes_column not in df.columns:
        raise ValueError(f"Clusters file: missing baseline column '{genome_allgenes_column}'")

    cluster_cols = [
        c
        for c in df.columns
        if c != genome_allgenes_column and c not in exclude_cluster_columns
    ]

    clusters: Dict[str, List[str]] = {}
    for c in cluster_cols:
        tags = df[c].dropna().astype(str).str.strip().tolist()
        tags = [t for t in tags if t and t.lower() != "nan"]
        seen = set()
        uniq: List[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        clusters[str(c)] = uniq

    allgenes = df[genome_allgenes_column].dropna().astype(str).str.strip().tolist()
    allgenes = [t for t in allgenes if t and t.lower() != "nan"]
    return clusters, allgenes



# -----------------------------
# score-mode helpers
# -----------------------------
def _get_devz_col(df: pd.DataFrame) -> str:
    for c in ["Mean_RCU_devZ", "RCU_devZ", "devZ", "DevZ"]:
        if c in df.columns:
            return c
    return ""


def _score_of_codons(df: pd.DataFrame, codons: List[str], *, agg: str = "sum") -> float:
    col = _get_devz_col(df)
    if not col:
        return np.nan
    sub = df[df["Codon"].isin(codons)][[col]].copy()
    vals = pd.to_numeric(sub[col], errors="coerce").values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals)) if agg.lower() == "mean" else float(np.sum(vals))


def _score_of_mask(df: pd.DataFrame, mask: pd.Series, *, agg: str = "sum") -> float:
    col = _get_devz_col(df)
    if not col:
        return np.nan
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce").values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals)) if agg.lower() == "mean" else float(np.sum(vals))


# -----------------------------
# heatmap matrix helpers
# -----------------------------
def _trna_prefix(trna_name: str) -> str:
    s = norm_str(trna_name)
    if not s:
        return ""
    import re
    m = re.match(r"([A-Za-z]+)", s)
    return m.group(1) if m else s


def _trna_sort_key(trna_name: str, prefix_order: Optional[List[str]] = None):
    import re
    s = norm_str(trna_name)
    pref = _trna_prefix(s)
    num = None
    m = re.search(r"(\d+)", s)
    if m:
        try:
            num = int(m.group(1))
        except Exception:
            num = None
    po = 10**9
    if prefix_order:
        try:
            po = prefix_order.index(pref)
        except ValueError:
            po = 10**9
    return (po, pref, num if num is not None else 10**9, s)


def _compute_trna_counts_from_df(df_counts: pd.DataFrame, codon_to_trna_set: Dict[str, Set[str]]) -> pd.Series:
    if df_counts is None or df_counts.empty:
        return pd.Series(dtype=float)
    out: Dict[str, float] = {}
    for _, r in df_counts.iterrows():
        cod = norm_str(r.get("Codon", ""))
        cnt = float(r.get("Count", 0.0)) if np.isfinite(r.get("Count", np.nan)) else 0.0
        for trna in codon_to_trna_set.get(cod, set()):
            out[trna] = out.get(trna, 0.0) + cnt
    return pd.Series(out).sort_index()


def choose_trna_row_order(
    base_df: pd.DataFrame,
    codon_to_trna_set: Dict[str, Set[str]],
    *,
    topn: int,
    aa_prefix_filter: List[str],
    keep_all_matches: bool,
    use_manual_order: bool = False,
    manual_order: Optional[List[str]] = None,
) -> List[str]:
    trna_counts = _compute_trna_counts_from_df(base_df, codon_to_trna_set)
    base_total = float(base_df["Count"].sum()) if ("Count" in base_df.columns) else float(trna_counts.sum())
    frac = (trna_counts / base_total) if base_total > 0 else trna_counts * np.nan
    frac = frac.fillna(0.0)

    # Manual order (fully explicit): keep only what exists in the baseline.
    if use_manual_order and manual_order:
        want = [canonicalize_trna_name(t) for t in manual_order]
        want = [t for t in want if norm_str(t)]
        if want:
            existing = set([canonicalize_trna_name(t) for t in frac.index])
            ordered = [t for t in want if t in existing]
            if ordered:
                return ordered
            # Explicit manual mode but no overlap: fail fast with a helpful message.
            sample = ", ".join(sorted(list(existing))[:20])
            raise ValueError(
                "plot_trna.manual_trna_order did not match any tRNA names found in the baseline. "
                f"Example available names: {sample}"
            )

    aa_pref = [p for p in aa_prefix_filter if norm_str(p)]
    if aa_pref:
        aa_pref_norm = [norm_str(x) for x in aa_pref]
        keep = []
        for t in frac.index:
            t0 = norm_str(t)
            pref = _trna_prefix(t0)
            if (pref in aa_pref_norm) or any(t0.startswith(pp) for pp in aa_pref_norm):
                keep.append(t)
        if keep:
            keep_sorted = sorted(keep, key=lambda x: _trna_sort_key(x, prefix_order=aa_pref_norm))
            if not keep_all_matches:
                top = frac.loc[keep_sorted].sort_values(ascending=False).head(topn).index.tolist()
                keep_sorted = [x for x in keep_sorted if x in set(top)]
            return keep_sorted

    return frac.sort_values(ascending=False).head(topn).index.tolist()


def choose_mod_row_order(
    *,
    site: str,
    base_df: pd.DataFrame,
    mod_by_codon: Dict[str, Set[str]],
    topn: int,
    row_selection_mode: str,
    score_mat_all: Optional[pd.DataFrame],
    use_manual_order: bool,
    manual_order: List[str],
    manual_order_append_automatic: bool,
):
    def frac_for_mod(df_in: pd.DataFrame, mod: str):
        tot = float(df_in["Count"].sum())
        if tot <= 0:
            return np.nan
        mask = df_in["Codon"].map(lambda c: mod in mod_by_codon.get(c, set()))
        return float(df_in.loc[mask, "Count"].sum() / tot)

    mods_all: List[str] = sorted({m for s in mod_by_codon.values() for m in s})

    if use_manual_order and manual_order:
        canon_to_actual = {canonical_mod(m): m for m in mods_all}
        chosen: List[str] = []
        seen = set()
        for want in manual_order:
            cw = canonical_mod(want)
            if not cw:
                continue
            if cw in canon_to_actual:
                m_actual = canon_to_actual[cw]
                if m_actual not in seen:
                    chosen.append(m_actual)
                    seen.add(m_actual)

        if manual_order_append_automatic:
            if row_selection_mode == "score_absmean" and score_mat_all is not None:
                rank = score_mat_all.abs().mean(axis=1).sort_values(ascending=False)
                auto_rank = [m for m in rank.index.tolist() if m in set(mods_all)]
            else:
                base_fracs = {m: frac_for_mod(base_df, m) for m in mods_all}
                auto_rank = sorted(mods_all, key=lambda m: (-(base_fracs.get(m, 0) if np.isfinite(base_fracs.get(m, 0)) else 0), m))
            for m in auto_rank:
                if m not in seen:
                    chosen.append(m)
                    seen.add(m)
                if len(chosen) >= max(topn, len(manual_order)):
                    break
        return chosen[:max(topn, len(chosen))]

    if row_selection_mode == "score_absmean" and score_mat_all is not None:
        rank = score_mat_all.abs().mean(axis=1).sort_values(ascending=False)
        return [m for m in rank.index.tolist() if m in set(mods_all)][:topn]

    base_fracs = {m: frac_for_mod(base_df, m) for m in mods_all}
    mods_ranked = sorted(mods_all, key=lambda m: (-(base_fracs.get(m, 0) if np.isfinite(base_fracs.get(m, 0)) else 0), m))
    return mods_ranked[:topn]


def _within_aa_usage_from_counts(df_in: pd.DataFrame, codon: str) -> float:
    aa = GENETIC_CODE_RNA.get(codon, "")
    if not aa or aa == "Stop":
        return np.nan
    sub = df_in[df_in["AA3"] == aa]
    tot = float(sub["Count"].sum())
    if tot <= 0:
        return 0.0
    if (df_in["Codon"] == codon).any():
        c = float(df_in.loc[df_in["Codon"] == codon, "Count"].values[0])
    else:
        c = 0.0
    return c / tot


def _within_box_usage_from_counts(df_in: pd.DataFrame, codon: str, box_codons: List[str]) -> float:
    if df_in is None or df_in.empty or not box_codons:
        return np.nan
    sub = df_in[df_in["Codon"].isin(box_codons)]
    tot = float(sub["Count"].sum()) if "Count" in sub.columns else 0.0
    if tot <= 0:
        return 0.0
    if (df_in["Codon"] == codon).any():
        c = float(df_in.loc[df_in["Codon"] == codon, "Count"].values[0])
    else:
        c = 0.0
    return c / tot


def _codon_has_trna(codon: str, trna_name: str, codon_to_trna_set: Dict[str, Set[str]]) -> bool:
    want = canonicalize_trna_name(trna_name)
    if not want:
        return False
    want_l = want.lower()
    return any(norm_str(tok).lower() == want_l for tok in codon_to_trna_set.get(codon, set()))


def _trna_label_for_codons(codons: List[str], codon_to_trna_set: Dict[str, Set[str]], fallback_label: str) -> str:
    sets = [set(codon_to_trna_set.get(c, set())) for c in codons]
    nonempty = [s for s in sets if len(s) > 0]
    if not nonempty:
        return fallback_label
    shared = set.intersection(*nonempty) if len(nonempty) >= 2 else nonempty[0]
    if shared:
        return "/".join(sorted(shared))
    uni = set().union(*nonempty)
    return "/".join(sorted(uni)) if uni else fallback_label


# -----------------------------
# main pipeline
# -----------------------------
def run_pipeline(cfg) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Accept either a raw dict (from the interactive runner) or an already-wrapped Cfg.
    cfg = cfg if isinstance(cfg, Cfg) else Cfg(cfg)

    mode = cfg.mode
    if mode not in {"fasta", "precomputed"}:
        raise ValueError("inputs.mode must be 'fasta' or 'precomputed'")

    out_mode = norm_str(cfg.get("analysis", "output_mode", default="enrichment")).lower()
    if out_mode not in {"enrichment", "score"}:
        raise ValueError("analysis.output_mode must be 'enrichment' or 'score'")

    # Paths (Codon-tRNAs decoding table)
    decoding_table_xlsx = cfg.path("inputs", "decoding_table_xlsx", required=False)
    if decoding_table_xlsx is None:
        # Backward-compatible key
        decoding_table_xlsx = cfg.path("inputs", "supertable_xlsx", required=True)

    if decoding_table_xlsx is None or not decoding_table_xlsx.exists():
        raise FileNotFoundError("Codon-tRNAs decoding table not found. Please select an existing .xlsx file.")

    decoding_table_sheet = norm_str(
        cfg.get("inputs", "decoding_table_sheet", default=cfg.get("inputs", "supertable_sheet", default="Long Format"))
    ) or "Long Format"

    # Common options
    exclude_aa = set([norm_str(x) for x in (cfg.get("reconstruction", "exclude_aa", default=["Met","Trp"]) or []) if norm_str(x)])
    beta = float(cfg.get("reconstruction", "beta", default=float(np.log(2.0))))
    z_agg = norm_str(cfg.get("reconstruction", "z_agg", default="mean")) or "mean"
    allow_aa_comp = bool(cfg.get("reconstruction", "allow_aa_composition_shift", default=False))
    aa_totals_use_gene_counts = bool(cfg.get("reconstruction", "aa_totals_use_cluster_gene_counts", default=True))
    pseudocount_per_gene = int(cfg.get("reconstruction", "pseudocount_per_gene", default=100))

    # Codon counting options
    counting_opts = CodonCountingOptions(
        drop_terminal_stop=bool(cfg.get("codon_counting", "drop_terminal_stop", default=True)),
        ignore_internal_stops=bool(cfg.get("codon_counting", "ignore_internal_stops", default=True)),
        include_ambiguous_codons=bool(cfg.get("codon_counting", "include_ambiguous_codons", default=False)),
    )

    # Outputs
    save_plots = cfg.save_plots
    show_plots = cfg.show_plots
    fig_fmt = cfg.figure_format
    fig_dpi = cfg.figure_dpi

    baseline_label = cfg.baseline_label
    cbar_label = cfg.colorbar_label

    # Plot toggles
    make_trna = bool(cfg.get("plots", "make_trna_shift_heatmap", default=True))
    make_mods = bool(cfg.get("plots", "make_mod_load_heatmaps", default=True))
    make_trna_box = bool(cfg.get("plots", "make_trna_box_wc_wobble_heatmap", default=True))
    make_genelevel = bool(cfg.get("plots", "make_genelevel_wc_wobble_supp", default=True))

    # Metric selection
    metric_trna = norm_str(cfg.get("plots", "trna_metric", default="rcu_devz")).lower()
    metric_mod34 = norm_str(cfg.get("plots", "mod34_metric", default="rcu_devz")).lower()
    metric_mod37 = norm_str(cfg.get("plots", "mod37_metric", default="rcu_devz")).lower()
    metric_trna_box = norm_str(cfg.get("plots", "trna_box_metric", default="rcu_devz")).lower()

    # Heatmap geometry and style
    geom = PubGeometryCm(
        cell_w=float(cfg.get("heatmap_geometry_cm", "cell_w", default=3.0)),
        cell_h=float(cfg.get("heatmap_geometry_cm", "cell_h", default=1.2)),
        left=float(cfg.get("heatmap_geometry_cm", "left", default=7.6)),
        right=float(cfg.get("heatmap_geometry_cm", "right", default=0.6)),
        top=float(cfg.get("heatmap_geometry_cm", "top", default=1.9)),
        bottom=float(cfg.get("heatmap_geometry_cm", "bottom", default=2.5)),
        cbar_w=float(cfg.get("heatmap_geometry_cm", "cbar_w", default=0.9)),
        cbar_pad=float(cfg.get("heatmap_geometry_cm", "cbar_pad", default=0.45)),
        cbar_extendfrac=float(cfg.get("heatmap_geometry_cm", "cbar_extendfrac", default=0.06)),
        left_group_bar_offset=float(cfg.get("heatmap_geometry_cm", "left_group_bar_offset", default=2.0)),
        left_group_label_offset=float(cfg.get("heatmap_geometry_cm", "left_group_label_offset", default=3.5)),
    )
    fonts = PubFontsPt(
        title=int(cfg.get("heatmap_fonts_pt", "title", default=18)),
        xtick=int(cfg.get("heatmap_fonts_pt", "xtick", default=16)),
        ytick=int(cfg.get("heatmap_fonts_pt", "ytick", default=20)),
        group_label=int(cfg.get("heatmap_fonts_pt", "group_label", default=26)),
        cbar_label=int(cfg.get("heatmap_fonts_pt", "cbar_label", default=24)),
        cbar_tick=int(cfg.get("heatmap_fonts_pt", "cbar_tick", default=20)),
    )
    style = HeatmapStyle(
        interpolation=norm_str(cfg.get("heatmap_style", "interpolation", default="nearest")) or "nearest",
        auto_scale=bool(cfg.get("heatmap_style", "auto_scale", default=True)),
        auto_clip_pct=tuple(cfg.get("heatmap_style", "auto_clip_pct", default=[1, 99])),
        auto_symmetric_around_zero=bool(cfg.get("heatmap_style", "auto_symmetric_around_zero", default=True)),
        colorbar_extend=norm_str(cfg.get("heatmap_style", "colorbar_extend", default="auto")) or "auto",
        invert_all_colormaps=bool(cfg.get("heatmap_style", "invert_all_colormaps", default=False)),
        x_label_wrap_len=int(cfg.get("heatmap_style", "x_label_wrap_len", default=9)),
        xtick_rotation=int(cfg.get("heatmap_style", "xtick_rotation", default=0)),
        ytick_rotation=int(cfg.get("heatmap_style", "ytick_rotation", default=0)),
    )
    force_same_cols = bool(cfg.get("heatmap_style", "force_same_columns_all_heatmaps", default=True))

    # Load Codon-tRNAs decoding table
    logger.info("[1/8] Loading Codon-tRNAs decoding table")
    codon_map, mod34_by_codon, mod37_by_codon, codon_to_trna_set, codon_trna_to_pairtype = load_decoding_table(
        decoding_table_xlsx, decoding_table_sheet
    )
    codon_to_pairtype_primary = {r["Codon"]: r["PairType"] for _, r in codon_map.iterrows()}

    seq_by_lt: Optional[Dict[str, str]] = None
    genome_gene_list: Optional[List[str]] = None
    clusters_order: List[str] = []
    tables_by_metric: Dict[str, Dict[str, pd.DataFrame]] = {}
    qc_by_metric: Dict[str, pd.DataFrame] = {}
    base_dir: Path

    # Always require CDS FASTA (used for FASTA mode and for precomputed baseline stability)
    cds_fasta = cfg.path("inputs", "cds_fasta", required=True)
    if cds_fasta is None or not cds_fasta.exists():
        raise FileNotFoundError("CDS FASTA not found. Please select an existing .fna/.fa/.fasta file.")
    seq_by_lt = read_fasta_cds(cds_fasta)

    if mode == "fasta":
        clusters_xlsx = cfg.path("inputs", "clusters_xlsx", required=True)
        if clusters_xlsx is None or not clusters_xlsx.exists():
            raise FileNotFoundError("Cluster workbook not found. Please select an existing .xlsx file.")
        base_dir = clusters_xlsx.parent
        clusters_sheet = norm_str(cfg.get("inputs", "clusters_sheet", default="Locus Tags")) or "Locus Tags"
        genome_col = norm_str(cfg.get("inputs", "genome_allgenes_column", default="Genome locus tags")) or "Genome locus tags"
        exclude_cols = cfg.get("inputs", "exclude_cluster_columns", default=[]) or []

        logger.info("[2/8] Loading clusters")
        clusters, allgenes = load_clusters(
            clusters_xlsx,
            sheet=clusters_sheet,
            genome_allgenes_column=genome_col,
            exclude_cluster_columns=list(exclude_cols),
        )
        clusters_order = ["All genes"] + list(clusters.keys())
        genome_gene_list = list(allgenes)

        logger.info("[3/8] Building codon tables from FASTA (RCU + RCU_devZ)")
        tables_by_metric, qc_by_metric = build_tables_from_fasta(
            clusters=clusters,
            allgenes=allgenes,
            codon_map=codon_map,
            seq_by_lt=seq_by_lt,
            opts=counting_opts,
            beta=beta,
            exclude_aa=exclude_aa,
            allow_aa_composition_shift=allow_aa_comp,
            aa_totals_use_cluster_gene_counts=aa_totals_use_gene_counts,
            pseudocount_per_gene=pseudocount_per_gene,
        )
        per_devz_info = None  # for genelevel in precomputed mode

    else:
        pre_rcu = cfg.path("inputs", "precomp_rcu_xlsx", required=True)
        if pre_rcu is None or not pre_rcu.exists():
            raise FileNotFoundError("Precomputed RCU workbook not found. Please select an existing .xlsx file.")
        pre_devz = cfg.path("inputs", "precomp_rcudevz_xlsx", required=True)
        if pre_devz is None or not pre_devz.exists():
            raise FileNotFoundError("Precomputed devZ workbook not found. Please select an existing .xlsx file.")
        base_dir = pre_devz.parent
        baseline_sheet = norm_str(cfg.get("inputs", "precomp_baseline_sheet", default="Genome locus tags")) or "Genome locus tags"
        exclude_sheets = cfg.get("inputs", "exclude_precomp_sheets", default=[]) or []

        logger.info("[2/8] Loading precomputed workbooks")
        baseline_rcu, per_rcu = load_precomputed_codonbias_workbook(
            pre_rcu, baseline_sheet=baseline_sheet, exclude_sheets=list(exclude_sheets), exclude_aa=exclude_aa
        )
        baseline_devz, per_devz = load_precomputed_codonbias_workbook(
            pre_devz, baseline_sheet=baseline_sheet, exclude_sheets=list(exclude_sheets), exclude_aa=exclude_aa
        )
        baseline_sheet_eff = baseline_sheet
        if baseline_sheet_eff not in per_rcu:
            baseline_sheet_eff = baseline_rcu
        if baseline_sheet_eff not in per_devz:
            baseline_sheet_eff = baseline_devz

        genome_gene_list = per_rcu.get(baseline_sheet_eff, {}).get("genes", None)

        logger.info("[3/8] Building codon tables from precomputed metrics")
        codon_tables_rcu, qc_rcu = build_tables_from_bias_precomputed(
            per_sheet=per_rcu,
            baseline_sheet=baseline_sheet_eff,
            codon_map=codon_map,
            seq_by_lt=seq_by_lt,
            opts=counting_opts,
            metric_kind="rcu",
            z_agg=z_agg,
            beta=beta,
            exclude_aa=exclude_aa,
            allow_aa_composition_shift=allow_aa_comp,
            aa_totals_use_cluster_gene_counts=aa_totals_use_gene_counts,
            pseudocount_per_gene=pseudocount_per_gene,
        )
        codon_tables_devz, qc_devz = build_tables_from_bias_precomputed(
            per_sheet=per_devz,
            baseline_sheet=baseline_sheet_eff,
            codon_map=codon_map,
            seq_by_lt=seq_by_lt,
            opts=counting_opts,
            metric_kind="rcu_devz",
            z_agg=z_agg,
            beta=beta,
            exclude_aa=exclude_aa,
            allow_aa_composition_shift=allow_aa_comp,
            aa_totals_use_cluster_gene_counts=aa_totals_use_gene_counts,
            pseudocount_per_gene=pseudocount_per_gene,
        )

        tables_by_metric = {"rcu": codon_tables_rcu, "rcu_devz": codon_tables_devz}
        qc_by_metric = {"rcu": qc_rcu, "rcu_devz": qc_devz}
        clusters_order = list(codon_tables_devz.keys())
        per_devz_info = per_devz
        clusters_xlsx = None

    # OUTPUT DIR
    if cfg.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{cfg.run_tag}" if cfg.run_tag else ""
        out_dir = (base_dir / "outputs" / f"run_{stamp}{tag}").resolve()
    else:
        out_dir = cfg.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # cluster master columns (exclude baseline)
    clusters_only_master = [c for c in clusters_order if c != "All genes"]

    # -----------------
    # Write codon tables
    # -----------------
    logger.info("[4/8] Writing codon tables (both metrics)")
    out_xlsx = out_dir / "01_CodonTables_byCluster__RCU_and_RCUdevZ.xlsx"
    used = set()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for metric_name in ["rcu", "rcu_devz"]:
            codon_tables = tables_by_metric.get(metric_name, {})
            qc_df = qc_by_metric.get(metric_name, None)

            for name, df in codon_tables.items():
                cols = [
                    "Codon", "AA3", "Count", "Freq", "RSCU", "RCU",
                    "Mean_RCU_devZ", "Mean_RCU",
                    "tRNAs", "Anticodon 5'-3'", "PairType", "Decoding",
                    "Mod34_atoms_str", "Mod37_atoms_str",
                    "tRMEs_str", "AllDecoders"
                ]
                cols = [c for c in cols if c in df.columns]
                sh = safe_sheetname(f"{metric_name}__{name}", used)
                df[cols].to_excel(writer, sheet_name=sh, index=False)

            if qc_df is not None and not qc_df.empty:
                sh = safe_sheetname(f"QC__{metric_name}", used)
                qc_df.to_excel(writer, sheet_name=sh, index=False)

    # -----------------
    # Supplementary: per-gene %WC vs %Wobble
    # -----------------
    if make_genelevel:
        logger.info("[5/8] Gene-level WC/Wobble supplementary outputs")
        gene_opt_raw = cfg.get("genelevel_wc_wobble", default={}) or {}
        gene_opts = GenelevelOptions(
            plot_kind=norm_str(gene_opt_raw.get("plot_kind", "box")).lower(),
            min_codons_per_gene=int(gene_opt_raw.get("min_codons_per_gene", 60)),
            include_allgenes=bool(gene_opt_raw.get("include_allgenes", True)),
            figsize_in=tuple(gene_opt_raw.get("figsize_in", [14, 5])),
            ylim=tuple(gene_opt_raw.get("ylim", [0, 100])) if gene_opt_raw.get("ylim", None) is not None else None,
            sort_by_mean=bool(gene_opt_raw.get("sort_by_mean", False)),
            errorbar_capsize=int(gene_opt_raw.get("errorbar_capsize", 3)),
            trnas_subset=list(gene_opt_raw.get("trnas_subset", [])),
            pairtype_rule=norm_str(gene_opt_raw.get("pairtype_rule", "any_wc")).lower(),
            x_wrap_len=int(style.x_label_wrap_len),
        )

        cluster_gene_lists: Dict[str, List[str]] = {}
        if gene_opts.include_allgenes and genome_gene_list is not None:
            cluster_gene_lists["All genes"] = list(genome_gene_list)

        if mode == "fasta":
            # reuse the clusters from file
            clusters, _all = load_clusters(
                clusters_xlsx,
                sheet=clusters_sheet,
                genome_allgenes_column=genome_col,
                exclude_cluster_columns=list(exclude_cols),
            )
            for k, v in clusters.items():
                cluster_gene_lists[k] = list(v)
        else:
            # precomputed: gene lists are sheet gene lists
            baseline_sheet_eff = norm_str(cfg.get("inputs", "precomp_baseline_sheet", default="Genome locus tags")) or "Genome locus tags"
            for sh, info in per_devz_info.items():
                if sh == baseline_sheet_eff:
                    continue
                cluster_gene_lists[sh] = list(info.get("genes", []))

        if cluster_gene_lists:
            trna_subset = [canonicalize_trna_name(x) for x in (gene_opts.trnas_subset or [])]
            trna_subset = [x for x in trna_subset if norm_str(x)]

            genelevel_pairtype_map, codons_allowed = build_codon_pairtype_map_for_trna_subset(
                trna_subset if trna_subset else None,
                codon_trna_to_pairtype,
                rule=gene_opts.pairtype_rule
            )

            per_gene_df, summary_df = compute_genelevel_wc_wobble_tables(
                cluster_gene_lists,
                seq_by_lt,
                genelevel_pairtype_map,
                counting_opts,
                min_codons_per_gene=gene_opts.min_codons_per_gene,
                codons_allowed=codons_allowed,
            )

            if per_gene_df is not None and not per_gene_df.empty:
                out_supp_xlsx = out_dir / "04_Supp_GeneLevel_WCvsWobble.xlsx"
                with pd.ExcelWriter(out_supp_xlsx, engine="openpyxl") as writer:
                    per_gene_df.to_excel(writer, sheet_name="PerGene", index=False)
                    if summary_df is not None and not summary_df.empty:
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)

                if summary_df is not None and not summary_df.empty and gene_opts.plot_kind in {"bar", "both"}:
                    plot_genelevel_bar(summary_df, plot_dir / f"Supp_GeneLevel_WCvsWobble__bar.{fig_fmt}",
                                       opts=gene_opts, save_plots=save_plots, show_plots=show_plots, figure_dpi=fig_dpi)
                if gene_opts.plot_kind in {"box", "both"}:
                    plot_genelevel_box(per_gene_df, plot_dir / f"Supp_GeneLevel_WCvsWobble__box.{fig_fmt}",
                                       clusters_order, opts=gene_opts, save_plots=save_plots, show_plots=show_plots, figure_dpi=fig_dpi)

    # -----------------
    # Build heatmap matrices
    # -----------------
    logger.info("[6/8] Building heatmap matrices")

    def get_tables_for_plot(plot_metric: str, plot_name: str):
        m = norm_str(plot_metric).lower()
        if out_mode == "score" and m != "rcu_devz":
            logger.warning("%s: output_mode='score' requires 'rcu_devz'. Falling back to rcu_devz.", plot_name)
            m = "rcu_devz"
        if m not in tables_by_metric:
            logger.warning("%s: metric '%s' not available. Falling back to 'rcu_devz'.", plot_name, m)
            m = "rcu_devz"
        return m, tables_by_metric[m]

    # Matrices to export
    trna_shift_wide = None
    trna_groups = None
    mod_heatmaps: Dict[str, pd.DataFrame] = {}
    trna_box_heatmap = None
    trna_box_groups = None

    # tRNA shift
    trna_metric_used, trna_tables = get_tables_for_plot(metric_trna, "tRNA plot")
    if make_trna:
        base_df = trna_tables["All genes"].copy()
        trna_cfg = cfg.get("plot_trna", default={}) or {}
        trna_row_order = choose_trna_row_order(
            base_df,
            codon_to_trna_set,
            topn=int(trna_cfg.get("topn", 50)),
            aa_prefix_filter=list(trna_cfg.get("aa_prefix_filter", [])),
            keep_all_matches=bool(trna_cfg.get("aa_filter_keep_all_matches", True)),
            use_manual_order=bool(trna_cfg.get("use_manual_trna_order", False)),
            manual_order=list(trna_cfg.get("manual_trna_order", []) or []),
        )

        if out_mode == "enrichment":
            base_counts = _compute_trna_counts_from_df(base_df, codon_to_trna_set)
            base_total = float(base_df["Count"].sum())
            base_frac = (base_counts / base_total) if base_total > 0 else base_counts * np.nan
            base_frac = base_frac.reindex(trna_row_order).fillna(0.0)

            mat = pd.DataFrame(index=trna_row_order, columns=clusters_only_master, dtype=float)
            use_log2 = bool(trna_cfg.get("use_log2", True))
            eps = float(trna_cfg.get("log_eps", 1e-12))
            for cl in clusters_only_master:
                dfc = trna_tables[cl].copy()
                cc = _compute_trna_counts_from_df(dfc, codon_to_trna_set)
                tot = float(dfc["Count"].sum())
                frac = (cc / tot) if tot > 0 else cc * np.nan
                frac = frac.reindex(trna_row_order).fillna(0.0)
                mat[cl] = np.log2((frac + eps) / (base_frac + eps)) if use_log2 else (frac - base_frac)
            trna_shift_wide = mat
        else:
            # score mode: sum/mean devZ over codons decoded by each tRNA
            codons_by_trna: Dict[str, List[str]] = {}
            for codon, trnas in codon_to_trna_set.items():
                for trna in trnas:
                    codons_by_trna.setdefault(trna, []).append(codon)
            mat = pd.DataFrame(index=trna_row_order, columns=clusters_only_master, dtype=float)
            for cl in clusters_only_master:
                dfc = trna_tables[cl]
                for trna in trna_row_order:
                    mat.loc[trna, cl] = _score_of_codons(dfc, codons_by_trna.get(trna, []), agg=norm_str(cfg.get("analysis","score_agg",default="sum")))
            trna_shift_wide = mat

        trna_groups = group_bounds_from_labels(list(trna_shift_wide.index), key_fn=_trna_prefix) if trna_shift_wide is not None else None

    # Mod heatmaps
    if make_mods:
        mod_cfg = cfg.get("plot_mods", default={}) or {}
        for site, plot_metric, mod_by_codon in [
            ("34", metric_mod34, mod34_by_codon),
            ("37", metric_mod37, mod37_by_codon),
        ]:
            metric_used, tables = get_tables_for_plot(plot_metric, f"Mods site {site}")
            base_df = tables["All genes"].copy()

            score_mat_all = None
            if out_mode == "score" and norm_str(mod_cfg.get("row_selection_mode","baseline_fraction")).lower() == "score_absmean":
                mods_all = sorted({m for s in mod_by_codon.values() for m in s})
                score_mat_all = pd.DataFrame(index=mods_all, columns=clusters_only_master, dtype=float)
                for cl in clusters_only_master:
                    dfc = tables[cl].copy()
                    for m in mods_all:
                        mask = dfc["Codon"].map(lambda c: m in mod_by_codon.get(c, set()))
                        score_mat_all.loc[m, cl] = _score_of_mask(dfc, mask, agg=norm_str(cfg.get("analysis","score_agg",default="sum")))

            manual_order = list(mod_cfg.get("manual_order_site37", DEFAULT_MOD37_MANUAL_ORDER)) if site == "37" else list(mod_cfg.get("manual_order_site34", DEFAULT_MOD34_MANUAL_ORDER))
            mods_show = choose_mod_row_order(
                site=site,
                base_df=base_df,
                mod_by_codon=mod_by_codon,
                topn=int(mod_cfg.get("topn", 30)),
                row_selection_mode=norm_str(mod_cfg.get("row_selection_mode","baseline_fraction")).lower(),
                score_mat_all=score_mat_all,
                use_manual_order=bool(mod_cfg.get("use_manual_order", True)),
                manual_order=manual_order,
                manual_order_append_automatic=bool(mod_cfg.get("manual_order_append_automatic", False)),
            )

            mat_site = pd.DataFrame(index=mods_show, columns=clusters_only_master, dtype=float)
            use_log2 = bool(mod_cfg.get("use_log2", True))
            eps = float(mod_cfg.get("log_eps", 1e-12))

            if out_mode == "enrichment":
                for cl in clusters_only_master:
                    dfc = tables[cl].copy()
                    tot_c = float(dfc["Count"].sum())
                    tot_a = float(base_df["Count"].sum())
                    for m in mods_show:
                        if tot_c <= 0 or tot_a <= 0:
                            mat_site.loc[m, cl] = np.nan
                            continue
                        mask_c = dfc["Codon"].map(lambda c: m in mod_by_codon.get(c, set()))
                        mask_a = base_df["Codon"].map(lambda c: m in mod_by_codon.get(c, set()))
                        fc = float(dfc.loc[mask_c, "Count"].sum() / tot_c)
                        fa = float(base_df.loc[mask_a, "Count"].sum() / tot_a)
                        mat_site.loc[m, cl] = np.log2((fc + eps) / (fa + eps)) if use_log2 else fc
            else:
                for cl in clusters_only_master:
                    dfc = tables[cl].copy()
                    for m in mods_show:
                        mask = dfc["Codon"].map(lambda c: m in mod_by_codon.get(c, set()))
                        mat_site.loc[m, cl] = _score_of_mask(dfc, mask, agg=norm_str(cfg.get("analysis","score_agg",default="sum")))
            mod_heatmaps[site] = mat_site

    # tRNA boxes heatmap
    if make_trna_box:
        box_cfg = cfg.get("plot_trna_box", default={}) or {}
        metric_used, tables = get_tables_for_plot(metric_trna_box, "tRNA-box plot")
        base_df = tables["All genes"].copy()

        if out_mode != "enrichment":
            logger.warning("tRNA-box heatmap is enrichment-only; skipping because output_mode='score'.")
        else:
            trnas_to_plot = [canonicalize_trna_name(t) for t in (box_cfg.get("trnas_to_plot", DEFAULT_TRNA_BOX_TRNAS) or DEFAULT_TRNA_BOX_TRNAS)]
            trnas_to_plot = [t for t in trnas_to_plot if norm_str(t)]

            use_log2 = bool(box_cfg.get("use_log2", True))
            eps = float(box_cfg.get("log_eps", 1e-12))
            sort_by_pairtype = bool(box_cfg.get("sort_by_pairtype", True))

            rows = []
            row_ticklabels = []
            row_group_trna = []

            def pt_rank(pt: str):
                pt = norm_str(pt)
                return 0 if pt == "WC" else (1 if pt == "Wobble" else 9)

            for trna in trnas_to_plot:
                box_codons = [c for c in SENSE_CODONS if _codon_has_trna(c, trna, codon_to_trna_set)]
                if not box_codons:
                    continue
                if sort_by_pairtype:
                    def pt_for(codon):
                        return norm_str(codon_trna_to_pairtype.get((codon, trna), codon_to_pairtype_primary.get(codon, "")))
                    box_codons = sorted(box_codons, key=lambda c: (pt_rank(pt_for(c)), c))
                else:
                    box_codons = sorted(box_codons)

                for codon in box_codons:
                    base_u = _within_box_usage_from_counts(base_df, codon, box_codons)
                    base_u = 0.0 if not np.isfinite(base_u) else base_u

                    row = {}
                    for cl in clusters_only_master:
                        dfc = tables[cl].copy()
                        u = _within_box_usage_from_counts(dfc, codon, box_codons)
                        u = 0.0 if not np.isfinite(u) else u
                        row[cl] = float(np.log2((u + eps) / (base_u + eps))) if use_log2 else float(u - base_u)

                    row_ticklabels.append(f"{codon}")
                    row_group_trna.append(trna)
                    rows.append(row)

            if rows:
                trna_box_heatmap = pd.DataFrame(rows, index=row_ticklabels)
                trna_box_groups = group_bounds_from_groupnames(row_group_trna)

    # enforce identical number of columns (fixed width)
    if force_same_cols:
        if trna_shift_wide is not None and not trna_shift_wide.empty:
            trna_shift_wide = pad_heatmap_columns(trna_shift_wide, clusters_only_master)
        for site in ["34", "37"]:
            if site in mod_heatmaps and mod_heatmaps[site] is not None and not mod_heatmaps[site].empty:
                mod_heatmaps[site] = pad_heatmap_columns(mod_heatmaps[site], clusters_only_master)
        if trna_box_heatmap is not None and not trna_box_heatmap.empty:
            trna_box_heatmap = pad_heatmap_columns(trna_box_heatmap, clusters_only_master)

    # -----------------
    # Plot heatmaps
    # -----------------
    logger.info("[7/8] Plotting heatmaps")
    if make_trna and trna_shift_wide is not None and not trna_shift_wide.empty:
        trna_cfg = cfg.get("plot_trna", default={}) or {}
        outp = plot_dir / f"Plot1_tRNA_shift__{trna_metric_used}.{fig_fmt}"
        plot_pub_heatmap(
            trna_shift_wide,
            title=f"tRNA usage shift (vs {baseline_label}) [{trna_metric_used}]",
            out_path=outp,
            cmap_name=norm_str(trna_cfg.get("cmap", "RdBu_r")),
            signed=True,
            geometry=geom,
            fonts=fonts,
            style=style,
            save_plots=save_plots,
            show_plots=show_plots,
            figure_dpi=fig_dpi,
            cbar_label=cbar_label,
            extend=style.colorbar_extend,
            vmin=float(trna_cfg.get("cbar_vmin", -1.5)),
            vmax=float(trna_cfg.get("cbar_vmax", 1.5)),
            abs_max=None if trna_cfg.get("cbar_abs_max", "") in {"", None} else float(trna_cfg.get("cbar_abs_max")),
            y_group_bounds=trna_groups,
            y_group_separator_style=norm_str(trna_cfg.get("group_separator_style", "black_box")),
            y_group_separator_linewidth=float(trna_cfg.get("group_separator_linewidth", 7.0)),
            y_group_separator_color=norm_str(trna_cfg.get("group_separator_color", "white")),
            y_group_box_color=norm_str(trna_cfg.get("group_box_color", "black")),
            y_group_box_linewidth=float(trna_cfg.get("group_box_linewidth", 2.0)),
            y_group_label_mode="center_left" if bool(trna_cfg.get("show_aa_group_labels", False)) else "none",
            y_group_link_bars=bool(trna_cfg.get("draw_group_link_bars", False)),
            y_group_link_style=norm_str(trna_cfg.get("group_link_style", "bracket")),
            y_group_link_fraction=float(trna_cfg.get("group_link_fraction", 0.33)),
            y_group_link_cap_len_axes=float(trna_cfg.get("group_link_cap_len_axes", 0.018)),
            y_group_link_endpad_rows=float(trna_cfg.get("group_link_endpad_rows", 0.15)),
        )

    if make_mods:
        mod_cfg = cfg.get("plot_mods", default={}) or {}
        for site in ["34", "37"]:
            mat = mod_heatmaps.get(site, None)
            if mat is None or mat.empty:
                continue
            outp = plot_dir / f"Plot2{'a' if site=='34' else 'b'}_Mods_site{site}.{fig_fmt}"
            vmin_s = float(mod_cfg.get("mod34_cbar_vmin", -0.3)) if site == "34" else float(mod_cfg.get("mod37_cbar_vmin", -0.3))
            vmax_s = float(mod_cfg.get("mod34_cbar_vmax", 0.3)) if site == "34" else float(mod_cfg.get("mod37_cbar_vmax", 0.3))
            plot_pub_heatmap(
                mat,
                title=f"Modification shift (site {site}) (vs {baseline_label})",
                out_path=outp,
                cmap_name=norm_str(mod_cfg.get("cmap", "RdBu_r")),
                signed=True,
                geometry=geom,
                fonts=fonts,
                style=style,
                save_plots=save_plots,
                show_plots=show_plots,
                figure_dpi=fig_dpi,
                cbar_label=cbar_label,
                extend=style.colorbar_extend,
                vmin=vmin_s,
                vmax=vmax_s,
                abs_max=None,
            )

    if make_trna_box and trna_box_heatmap is not None and not trna_box_heatmap.empty:
        box_cfg = cfg.get("plot_trna_box", default={}) or {}
        outp = plot_dir / f"Plot3_tRNAboxes_WCvsWobble__{metric_trna_box}.{fig_fmt}"
        plot_pub_heatmap(
            trna_box_heatmap,
            title=f"Decoding in tRNA boxes: WC vs Wobble (vs {baseline_label}) [{metric_trna_box}]",
            out_path=outp,
            cmap_name=norm_str(box_cfg.get("cmap", "RdBu_r")),
            signed=True,
            geometry=geom,
            fonts=fonts,
            style=style,
            save_plots=save_plots,
            show_plots=show_plots,
            figure_dpi=fig_dpi,
            cbar_label=cbar_label,
            extend=style.colorbar_extend,
            vmin=float(box_cfg.get("cbar_vmin", -1.5)),
            vmax=float(box_cfg.get("cbar_vmax", 1.5)),
            abs_max=None,
            y_group_bounds=trna_box_groups,
            y_group_separator_style=norm_str(box_cfg.get("group_separator_style", "black_box")),
            y_group_separator_linewidth=float(box_cfg.get("group_separator_linewidth", 7.0)),
            y_group_separator_color=norm_str(box_cfg.get("group_separator_color", "white")),
            y_group_box_color=norm_str(box_cfg.get("group_box_color", "black")),
            y_group_box_linewidth=float(box_cfg.get("group_box_linewidth", 2.0)),
            y_group_label_mode="center_left" if bool(box_cfg.get("show_trna_group_labels", True)) else "none",
            y_group_link_bars=bool(box_cfg.get("draw_group_link_bars", False)),
            y_group_link_style=norm_str(box_cfg.get("group_link_style", "bracket")),
            y_group_link_fraction=float(box_cfg.get("group_link_fraction", 0.33)),
            y_group_link_cap_len_axes=float(box_cfg.get("group_link_cap_len_axes", 0.018)),
            y_group_link_endpad_rows=float(box_cfg.get("group_link_endpad_rows", 0.15)),
        )

    # -----------------
    # Write heatmap matrices
    # -----------------
    logger.info("[8/8] Writing heatmap matrices")
    out_mat = out_dir / "02_HeatmapMatrices.xlsx"
    used2 = set()
    wrote_any = False

    with pd.ExcelWriter(out_mat, engine="openpyxl") as writer:
        if trna_shift_wide is not None and not trna_shift_wide.empty:
            trna_shift_wide.to_excel(
                writer,
                sheet_name=safe_sheetname(f"tRNA_shift__{trna_metric_used}", used2),
            )
            wrote_any = True

        for site in ["34", "37"]:
            if site in mod_heatmaps and mod_heatmaps[site] is not None and not mod_heatmaps[site].empty:
                mod_heatmaps[site].to_excel(
                    writer,
                    sheet_name=safe_sheetname(f"Mods_site{site}", used2),
                )
                wrote_any = True

        if trna_box_heatmap is not None and not trna_box_heatmap.empty:
            trna_box_heatmap.to_excel(
                writer,
                sheet_name=safe_sheetname("tRNA_boxes_WC_vs_Wobble", used2),
            )
            wrote_any = True

        # openpyxl requires at least one visible sheet.
        if not wrote_any:
            pd.DataFrame(
                {
                    "Note": [
                        "No heatmap matrices were generated.",
                        "This typically happens when all plot/matrix toggles are disabled, or when the selected filters yield no data.",
                        "Enable at least one of: plot_trna / plot_mods / plot_trna_box, or relax filters (e.g., topn, AA filters, min counts).",
                    ]
                }
            ).to_excel(
                writer,
                sheet_name=safe_sheetname("README", used2),
                index=False,
            )

    logger.info("Done. Output folder: %s", out_dir)
    return out_dir