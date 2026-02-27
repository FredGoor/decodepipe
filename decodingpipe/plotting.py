from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import cm_to_in, norm_str, wrap_label_no_break
from .fasta import CodonCountingOptions, count_codons_in_seq


# =========================
# Heatmap plotting
# =========================


@dataclass(frozen=True)
class PubGeometryCm:
    cell_w: float
    cell_h: float
    left: float
    right: float
    top: float
    bottom: float
    cbar_w: float
    cbar_pad: float
    cbar_extendfrac: float
    # Offsets to the left of y tick labels
    left_group_bar_offset: float
    left_group_label_offset: float


@dataclass(frozen=True)
class PubFontsPt:
    title: int
    xtick: int
    ytick: int
    group_label: int
    cbar_label: int
    cbar_tick: int


@dataclass(frozen=True)
class HeatmapStyle:
    interpolation: str = "nearest"
    auto_scale: bool = True
    auto_clip_pct: Tuple[float, float] = (1, 99)
    auto_symmetric_around_zero: bool = True
    colorbar_extend: str = "auto"
    invert_all_colormaps: bool = False
    x_label_wrap_len: int = 9
    xtick_rotation: int = 0
    ytick_rotation: int = 0


def resolve_cmap(cmap_name: str, invert: bool):
    cmap = plt.get_cmap(cmap_name if norm_str(cmap_name) else "viridis")
    return cmap.reversed() if invert else cmap


def _auto_limits_from_data(data: np.ndarray, clip_pct: Tuple[float, float] | None):
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return None, None
    if clip_pct is None:
        return float(np.min(vals)), float(np.max(vals))
    lo = float(np.percentile(vals, clip_pct[0]))
    hi = float(np.percentile(vals, clip_pct[1]))
    return lo, hi


def resolve_heatmap_limits(
    matrix_df: pd.DataFrame,
    *,
    signed: bool,
    vmin: float | None,
    vmax: float | None,
    abs_max: float | None,
    style: HeatmapStyle,
):
    if abs_max is not None:
        m = float(abs_max)
        return (-m, m) if signed else (0.0, m)
    if (vmin is not None) or (vmax is not None):
        return vmin, vmax
    if not style.auto_scale:
        return None, None

    data = np.array(matrix_df.values, dtype=float)
    lo, hi = _auto_limits_from_data(data, clip_pct=style.auto_clip_pct)
    if lo is None:
        return None, None
    if signed and style.auto_symmetric_around_zero:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def pad_heatmap_columns(df: pd.DataFrame, master_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.reindex(columns=master_cols)


def plot_pub_heatmap(
    matrix_df: pd.DataFrame,
    *,
    title: str,
    out_path: Optional[Path],
    cmap_name: str,
    signed: bool,
    geometry: PubGeometryCm,
    fonts: PubFontsPt,
    style: HeatmapStyle,
    save_plots: bool,
    show_plots: bool,
    figure_dpi: int,
    cbar_label: str,
    extend: str = "auto",
    vmin: float | None = None,
    vmax: float | None = None,
    abs_max: float | None = None,
    ytick_labels: Sequence[str] | None = None,
    y_group_bounds: List[Tuple[int, int, str]] | None = None,
    y_group_separator_style: str = "none",  # "none" | "white_gap" | "black_line" | "black_box"
    y_group_separator_linewidth: float = 6.0,
    y_group_separator_color: str = "white",
    y_group_box_color: str = "black",
    y_group_box_linewidth: float = 1.2,
    y_group_label_mode: str = "none",  # "none" | "center_left"
    # New: draw a vertical group-link bar between y tick labels and left group labels
    y_group_link_bars: bool = False,
    y_group_link_style: str = "bracket",
    y_group_link_fraction: float = 0.33,
    y_group_link_cap_len_axes: float = 0.018,
    # Shrink each per-group marker at both ends (in *row units*) so markers do not touch
    # across adjacent groups. This is what creates a clear visual "gap" between braces.
    y_group_link_endpad_rows: float = 0.15,
    y_group_link_bar_color: str = "black",
    y_group_link_bar_linewidth: float = 2.2,
):
    if matrix_df is None or matrix_df.empty:
        return

    # labels
    xlabels = list(matrix_df.columns)
    if style.x_label_wrap_len:
        xlabels = [wrap_label_no_break(x, style.x_label_wrap_len) for x in xlabels]
    ylabels = list(matrix_df.index) if ytick_labels is None else list(ytick_labels)

    nrows, ncols = len(ylabels), len(xlabels)

    vmin_eff, vmax_eff = resolve_heatmap_limits(
        matrix_df, signed=signed, vmin=vmin, vmax=vmax, abs_max=abs_max, style=style
    )

    data = np.array(matrix_df.values, dtype=float)
    mask = ~np.isfinite(data)
    data_ma = np.ma.array(data, mask=mask)

    # colorbar extend
    if extend == "auto" and (vmin_eff is not None) and (vmax_eff is not None):
        dmin = np.nanmin(data) if np.isfinite(data).any() else vmin_eff
        dmax = np.nanmax(data) if np.isfinite(data).any() else vmax_eff
        extend_eff = "both" if (dmin < vmin_eff or dmax > vmax_eff) else "neither"
    else:
        extend_eff = extend if extend in {"neither", "both", "min", "max"} else "neither"

    # geometry (cm -> inches)
    cell_w_in = cm_to_in(geometry.cell_w)
    cell_h_in = cm_to_in(geometry.cell_h)

    left_in = cm_to_in(geometry.left)
    right_in = cm_to_in(geometry.right)
    top_in = cm_to_in(geometry.top)
    bottom_in = cm_to_in(geometry.bottom)

    cbar_w_in = cm_to_in(geometry.cbar_w)
    cbar_pad_in = cm_to_in(geometry.cbar_pad)

    heat_w_in = cell_w_in * max(1, ncols)
    heat_h_in = cell_h_in * max(1, nrows)

    fig_w_in = left_in + heat_w_in + cbar_pad_in + cbar_w_in + right_in
    fig_h_in = bottom_in + heat_h_in + top_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor("white")

    ax_left = left_in / fig_w_in
    ax_bottom = bottom_in / fig_h_in
    ax_w = heat_w_in / fig_w_in
    ax_h = heat_h_in / fig_h_in

    cax_left = (left_in + heat_w_in + cbar_pad_in) / fig_w_in
    cax_w = cbar_w_in / fig_w_in

    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
    cax = fig.add_axes([cax_left, ax_bottom, cax_w, ax_h])
    ax.set_facecolor("white")

    cmap = resolve_cmap(cmap_name, invert=style.invert_all_colormaps)
    im = ax.imshow(
        data_ma,
        aspect="auto",
        interpolation=style.interpolation,
        cmap=cmap,
        vmin=vmin_eff,
        vmax=vmax_eff,
    )

    cb = fig.colorbar(im, cax=cax, extend=extend_eff, extendfrac=geometry.cbar_extendfrac)
    cb.set_label(cbar_label, fontsize=fonts.cbar_label)
    cb.ax.tick_params(labelsize=fonts.cbar_tick)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=style.xtick_rotation, fontsize=fonts.xtick)
    for lab in ax.get_xticklabels():
        lab.set_ha("center")
        lab.set_va("top")

    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, rotation=style.ytick_rotation, fontsize=fonts.ytick)
    ax.set_title(title, fontsize=fonts.title, pad=10)

    # group separators / boxes
    if y_group_bounds and norm_str(y_group_separator_style).lower() != "none":
        style_sep = y_group_separator_style.lower()

        if style_sep == "black_box":
            import matplotlib.patches as patches
            for (s, e, _) in y_group_bounds:
                rect = patches.Rectangle(
                    (-0.5, s - 0.5),
                    width=len(xlabels),
                    height=(e - s + 1),
                    fill=False,
                    edgecolor=y_group_box_color,
                    linewidth=y_group_box_linewidth,
                    zorder=5,
                )
                ax.add_patch(rect)

        if style_sep in {"white_gap", "black_line"}:
            col = y_group_separator_color if style_sep == "white_gap" else "black"
            lw = float(y_group_separator_linewidth)
            for i in range(len(y_group_bounds) - 1):
                boundary = y_group_bounds[i][1] + 0.5
                ax.hlines(boundary, xmin=-0.5, xmax=len(xlabels) - 0.5, colors=col, linewidth=lw, zorder=6)

    # group labels on left + optional group-link bars
    if y_group_bounds and norm_str(y_group_label_mode).lower() == "center_left":
        # Convert cm offsets to x positions in *axes* coordinates (negative = left of axis)
        label_pad_axes = (cm_to_in(geometry.left_group_label_offset) / heat_w_in) if heat_w_in > 0 else 0.08
        lab_x = -label_pad_axes

        # Group-link bracket position:
        # by default, place it at a fixed fraction of the distance between the axis (x=0)
        # and the right edge of the group label (lab_x). For example, 0.33 is "one third".
        try:
            frac = float(y_group_link_fraction)
        except Exception:
            frac = None

        if frac is None:
            bar_pad_axes = (cm_to_in(geometry.left_group_bar_offset) / heat_w_in) if heat_w_in > 0 else (0.05 * max(1.0, label_pad_axes))
            bar_x = -bar_pad_axes
        else:
            bar_x = lab_x * float(frac)

        # Helper: a true curly brace path patch (per group)
        def _add_curly_brace(
            *,
            x_axes: float,
            y0_data: float,
            y1_data: float,
            width_axes: float,
            color: str,
            lw: float,
        ):
            """Draw a left-side curly brace ("{") spanning [y0_data, y1_data].

            Coordinates use ax.get_yaxis_transform(): x in axes coordinates, y in data.
            """
            from matplotlib.path import Path
            import matplotlib.patches as patches

            y0 = float(y0_data)
            y1 = float(y1_data)
            if y1 <= y0:
                return
            dy = y1 - y0
            ym = 0.5 * (y0 + y1)
            w = float(width_axes)

            # Control points are expressed as fractions of dy.
            # This produces a brace that reads well across small and large groups.
            verts = [
                (x_axes, y0),
                (x_axes, y0 + 0.10 * dy),
                (x_axes + w, y0 + 0.18 * dy),
                (x_axes + w, y0 + 0.32 * dy),

                (x_axes + w, y0 + 0.46 * dy),
                (x_axes, ym - 0.04 * dy),
                (x_axes, ym),

                (x_axes, ym + 0.04 * dy),
                (x_axes + w, y0 + 0.54 * dy),
                (x_axes + w, y0 + 0.68 * dy),

                (x_axes + w, y0 + 0.82 * dy),
                (x_axes, y0 + 0.90 * dy),
                (x_axes, y1),
            ]
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
            ]

            patch = patches.PathPatch(
                Path(verts, codes),
                fill=False,
                edgecolor=color,
                linewidth=float(lw),
                capstyle="butt",
                joinstyle="miter",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                zorder=7,
            )
            ax.add_patch(patch)

        for (s, e, lab) in y_group_bounds:
            yc = 0.5 * (s + e)

            if y_group_link_bars:
                # Shrink at both ends so adjacent braces do not touch
                y0 = float(s - 0.5)
                y1 = float(e + 0.5)
                pad = float(y_group_link_endpad_rows)
                # Keep a minimum visible brace height
                pad = max(0.0, min(pad, 0.45 * max(1.0, (y1 - y0))))
                y0 += pad
                y1 -= pad
                style_link = norm_str(y_group_link_style).lower() or "bracket"

                if style_link in {"brace", "curly", "curly_brace"}:
                    # True curly brace spanning the group
                    _add_curly_brace(
                        x_axes=float(bar_x),
                        y0_data=float(y0),
                        y1_data=float(y1),
                        width_axes=float(y_group_link_cap_len_axes),
                        color=y_group_link_bar_color,
                        lw=float(y_group_link_bar_linewidth),
                    )
                else:
                    # Bracket-like marker (per group)
                    ax.plot(
                        [bar_x, bar_x],
                        [y0, y1],
                        transform=ax.get_yaxis_transform(),
                        color=y_group_link_bar_color,
                        linewidth=float(y_group_link_bar_linewidth),
                        solid_capstyle="butt",
                        clip_on=False,
                        zorder=7,
                    )

                    cap = float(y_group_link_cap_len_axes)
                    ax.plot(
                        [bar_x, bar_x + cap],
                        [y0, y0],
                        transform=ax.get_yaxis_transform(),
                        color=y_group_link_bar_color,
                        linewidth=float(y_group_link_bar_linewidth),
                        solid_capstyle="butt",
                        clip_on=False,
                        zorder=7,
                    )
                    ax.plot(
                        [bar_x, bar_x + cap],
                        [y1, y1],
                        transform=ax.get_yaxis_transform(),
                        color=y_group_link_bar_color,
                        linewidth=float(y_group_link_bar_linewidth),
                        solid_capstyle="butt",
                        clip_on=False,
                        zorder=7,
                    )

            ax.text(
                lab_x,
                yc,
                str(lab),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=fonts.group_label,
            )

    if save_plots and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=figure_dpi)
    if show_plots:
        plt.show()
    plt.close(fig)


# =========================
# Gene-level supplementary plot
# =========================


@dataclass(frozen=True)
class GenelevelOptions:
    plot_kind: str = "box"          # "bar" | "box" | "both"
    min_codons_per_gene: int = 60
    include_allgenes: bool = True
    figsize_in: Tuple[float, float] = (14, 5)
    ylim: Tuple[float, float] | None = (0, 100)
    sort_by_mean: bool = False
    errorbar_capsize: int = 3
    trnas_subset: List[str] | None = None
    pairtype_rule: str = "any_wc"   # "any_wc" | "wc_only" | "any_wobble"
    x_wrap_len: int = 9


def build_codon_pairtype_map_for_trna_subset(
    trna_subset: Optional[List[str]],
    codon_trna_to_pairtype: Dict[Tuple[str, str], str],
    *,
    rule: str = "any_wc",
):
    from .utils import canonicalize_trna_name

    trnas = [canonicalize_trna_name(t) for t in (trna_subset or [])]
    trnas = [t for t in trnas if norm_str(t)]
    trna_set = set(trnas)

    per_codon_pts: Dict[str, Set[str]] = {}
    for (cod, trna), pt in codon_trna_to_pairtype.items():
        if trna_set and trna not in trna_set:
            continue
        per_codon_pts.setdefault(cod, set()).add(norm_str(pt) if norm_str(pt) else "Wobble")

    rule = norm_str(rule).lower()
    codon_to_pt: Dict[str, str] = {}

    for cod, pts in per_codon_pts.items():
        pts = set([p if p in {"WC", "Wobble"} else "Wobble" for p in pts])
        if rule == "wc_only":
            codon_to_pt[cod] = "WC" if pts == {"WC"} else "Wobble"
        elif rule == "any_wobble":
            codon_to_pt[cod] = "Wobble" if "Wobble" in pts else "WC"
        else:
            codon_to_pt[cod] = "WC" if "WC" in pts else "Wobble"

    return codon_to_pt, set(codon_to_pt.keys())


def _gene_wc_wobble_stats_from_seq(
    dna_seq: str,
    codon_to_pairtype: Dict[str, str],
    opts: CodonCountingOptions,
    *,
    codons_allowed: Optional[Set[str]] = None,
):
    counts = count_codons_in_seq(dna_seq, opts)
    wc = wob = ignored = 0

    for codon, n in counts.items():
        if codons_allowed is not None and codon not in codons_allowed:
            ignored += int(n)
            continue
        pt = codon_to_pairtype.get(codon, None)
        if pt is None:
            ignored += int(n)
            continue
        if norm_str(pt) == "WC":
            wc += int(n)
        else:
            wob += int(n)

    total = wc + wob
    if total <= 0:
        return {
            "wc_codons": 0,
            "wobble_codons": 0,
            "total_codons": 0,
            "pct_wc": np.nan,
            "pct_wobble": np.nan,
            "codons_ignored": int(ignored),
        }
    pct_wc = 100.0 * (wc / float(total))
    return {
        "wc_codons": int(wc),
        "wobble_codons": int(wob),
        "total_codons": int(total),
        "pct_wc": float(pct_wc),
        "pct_wobble": float(100.0 - pct_wc),
        "codons_ignored": int(ignored),
    }


def compute_genelevel_wc_wobble_tables(
    cluster_gene_lists: Dict[str, List[str]],
    seq_by_lt: Dict[str, str],
    codon_to_pairtype: Dict[str, str],
    opts: CodonCountingOptions,
    *,
    min_codons_per_gene: int,
    codons_allowed: Optional[Set[str]] = None,
):
    all_genes_union: List[str] = []
    seen = set()
    for genes in cluster_gene_lists.values():
        for g in genes:
            g = norm_str(g)
            if g and g not in seen:
                seen.add(g)
                all_genes_union.append(g)

    per_gene_rows = []
    for g in all_genes_union:
        seq = seq_by_lt.get(g)
        if seq is None:
            continue
        st = _gene_wc_wobble_stats_from_seq(seq, codon_to_pairtype, opts, codons_allowed=codons_allowed)
        if min_codons_per_gene and np.isfinite(st["total_codons"]):
            if int(st["total_codons"]) < int(min_codons_per_gene):
                continue
        per_gene_rows.append({"Gene": g, **st})

    if not per_gene_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_gene_base = pd.DataFrame(per_gene_rows)

    out_rows = []
    for cl, genes in cluster_gene_lists.items():
        genes_set = set([norm_str(x) for x in genes if norm_str(x)])
        sub = per_gene_base[per_gene_base["Gene"].isin(genes_set)].copy()
        if sub.empty:
            continue
        sub.insert(0, "Cluster", cl)
        out_rows.append(sub)

    if not out_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_gene_df = pd.concat(out_rows, ignore_index=True)

    summ = []
    for cl, sub in per_gene_df.groupby("Cluster"):
        vals = pd.to_numeric(sub["pct_wc"], errors="coerce").values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        summ.append(
            {
                "Cluster": cl,
                "n_genes": int(vals.size),
                "mean_pct_wc": float(np.mean(vals)),
                "sd_pct_wc": float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0,
                "median_pct_wc": float(np.median(vals)),
            }
        )

    summary_df = pd.DataFrame(summ)
    return per_gene_df, summary_df


def plot_genelevel_bar(
    summary_df: pd.DataFrame,
    out_path: Optional[Path],
    *,
    opts: GenelevelOptions,
    save_plots: bool,
    show_plots: bool,
    figure_dpi: int,
):
    if summary_df is None or summary_df.empty:
        return

    df = summary_df.copy()
    if opts.sort_by_mean:
        df = df.sort_values("mean_pct_wc", ascending=False)

    x = df["Cluster"].tolist()
    y = df["mean_pct_wc"].tolist()
    e = df["sd_pct_wc"].tolist()
    xlab = [wrap_label_no_break(s, opts.x_wrap_len) for s in x]

    fig, ax = plt.subplots(figsize=opts.figsize_in)
    ax.bar(range(len(x)), y, yerr=e, capsize=opts.errorbar_capsize)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(xlab, rotation=0)
    ax.set_ylabel("Mean % codons decoded by WC (per gene)")
    ax.set_title("Supplementary: per-gene WC vs Wobble decoding (cluster means ± SD)")
    if opts.ylim is not None:
        ax.set_ylim(opts.ylim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_plots and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=figure_dpi)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_genelevel_box(
    per_gene_df: pd.DataFrame,
    out_path: Optional[Path],
    clusters_order: List[str] | None,
    *,
    opts: GenelevelOptions,
    save_plots: bool,
    show_plots: bool,
    figure_dpi: int,
):
    if per_gene_df is None or per_gene_df.empty:
        return

    df = per_gene_df.copy()
    clusters = sorted(df["Cluster"].unique().tolist())
    if clusters_order:
        clusters = [c for c in clusters_order if c in set(clusters)] or clusters

    if opts.sort_by_mean:
        means = df.groupby("Cluster")["pct_wc"].mean().to_dict()
        clusters = sorted(clusters, key=lambda c: -float(means.get(c, -np.inf)))

    data = []
    labels = []
    for cl in clusters:
        vals = pd.to_numeric(df.loc[df["Cluster"] == cl, "pct_wc"], errors="coerce").values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(wrap_label_no_break(cl, opts.x_wrap_len))

    if not data:
        return

    fig, ax = plt.subplots(figsize=opts.figsize_in)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("% codons decoded by WC (per gene)")
    ax.set_title("Supplementary: per-gene WC vs Wobble decoding (distributions)")
    if opts.ylim is not None:
        ax.set_ylim(opts.ylim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_plots and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=figure_dpi)
    if show_plots:
        plt.show()
    plt.close(fig)
