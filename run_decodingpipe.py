#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""decodingpipe — codon-cluster decoding strategy analysis

This entry script is intentionally *interactive* and meant for day-to-day use.
It launches file pickers and then runs the pipeline.

The pipeline expects a **Codon-tRNAs decoding table** (Excel) describing codon↔tRNA
pairings and associated modifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from decodingpipe.pipeline import run_pipeline


# =========================
# === USER SETTINGS      ===
# =========================

# Default input mode suggested in the interactive prompt: "fasta" or "precomputed"
DEFAULT_INPUT_MODE = "fasta"

# Plot output behavior
SHOW_PLOTS = True
SAVE_PLOTS = True
FIGURE_FORMAT = "png"   # "png" or "pdf"
FIGURE_DPI = 220

# Plot toggles (set True/False here to enable/disable each plot)
MAKE_TRNA_SHIFT_HEATMAP = True
MAKE_MOD_LOAD_HEATMAPS = True
MAKE_TRNA_BOX_WC_WOBBLE_HEATMAP = True
MAKE_GENELEVEL_WC_WOBBLE_SUPP = True


# =========================
# === PLOT SPECIFICATIONS ===
# =========================

# --- (A) tRNA usage shift heatmap (row order) ---
# Requested canonical order (edit freely):
TRNA_SHIFT_TRNAS_ORDER = [
    "Arg1", "Arg2", "Arg3", "Arg4",
    "Gly1", "Gly2", "Gly3",
    "Ile1", "Ile2",
    "Leu1", "Leu2", "Leu3", "Leu4", "Leu5",
    "Pro1", "Pro2", "Pro3",
    "Ser1", "Ser2", "Ser3", "Ser4",
]

# Show amino-acid group labels (Arg, Gly, Ile, ...) on the far left
TRNA_SHIFT_SHOW_AA_GROUP_LABELS = True


# --- (C) WC vs Wobble heatmap within tRNA boxes ---
# (Rows will be codons, grouped by these tRNA boxes; edit freely.)
TRNA_BOX_TRNAS_TO_PLOT = [
    "Arg1", "Asn1", "Asp1", "Cys1", "His1", "Ile1", "Phe1", "Tyr1",
]

# --- (D) Heatmap typography / geometry (cm and pt) ---
# Centralized here so you do not need to hunt through modules.
HEATMAP_GEOMETRY_CM = {
    "cell_w": 3.0,
    "cell_h": 1.2,
    "left": 7.6,
    "right": 0.6,
    "top": 1.9,
    "bottom": 2.5,
    "cbar_w": 0.9,
    "cbar_pad": 0.45,
    # Group label and group-link bar offsets (to the left of y tick labels)
    "left_group_bar_offset": 2.0,
    "left_group_label_offset": 3.5,
}

HEATMAP_FONTS_PT = {
    "title": 18,
    "xtick": 16,
    "ytick": 20,
    "group_label": 26,
    "cbar_label": 24,
    "cbar_tick": 20,
}

HEATMAP_STYLE = {
    "interpolation": "nearest",
    "auto_scale": True,
    "auto_clip_pct": [1, 99],
    "auto_symmetric_around_zero": True,
    "colorbar_extend": "auto",
    "invert_all_colormaps": False,
    "x_label_wrap_len": 9,
    "xtick_rotation": 0,
    "ytick_rotation": 0,
    # Make all heatmaps share the same columns (same width), for clean panels
    "force_same_columns_all_heatmaps": True,
}

# Analysis output mode
# - "enrichment" (log2 fold-change vs baseline)  [recommended]
# - "score"      (devZ-based cluster scores)
OUTPUT_MODE = "enrichment"
SCORE_AGG = "sum"  # only used when OUTPUT_MODE == "score" ("sum" or "mean")

# Labels
BASELINE_LABEL = "Genome"
COLORBAR_LABEL = "Log2 fold change vs baseline"


def _pick_mode_tk(default: str = "fasta") -> str:
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    try:
        mode = simpledialog.askstring(
            title="decodingpipe",
            prompt="Choose input mode: 'fasta' or 'precomputed'",
            initialvalue=default,
            parent=root,
        )
    finally:
        root.destroy()

    mode = (mode or "").strip().lower()
    return mode if mode in {"fasta", "precomputed"} else (default or "fasta")


def _ask_sheet_tk(xlsx_path: Path, title: str, default: Optional[str] = None) -> str:
    import tkinter as tk
    from tkinter import simpledialog

    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names
    if len(sheets) == 1:
        return sheets[0]

    root = tk.Tk()
    root.withdraw()
    try:
        prompt = (
            f"{title}: enter the sheet name to use.\n\nAvailable sheets:\n"
            + "\n".join(f"  - {s}" for s in sheets)
        )
        sheet = simpledialog.askstring(
            title="decodingpipe",
            prompt=prompt,
            initialvalue=default or sheets[0],
            parent=root,
        )
    finally:
        root.destroy()

    sheet = (sheet or "").strip()
    return sheet if sheet in sheets else sheets[0]


def _guess_baseline_column(colnames) -> str:
    cols = [str(c).strip() for c in colnames]
    if not cols:
        return ""
    low = [c.lower() for c in cols]

    candidates = [
        "genome locus tags",
        "genome",
        "all genes",
        "allgenes",
        "all",
        "baseline",
    ]
    for want in candidates:
        for c, cl in zip(cols, low):
            if want == cl:
                return c
    for c, cl in zip(cols, low):
        if "genome" in cl and "tag" in cl:
            return c
    for c, cl in zip(cols, low):
        if "all" in cl and "gene" in cl:
            return c
    return cols[0]


def _ask_baseline_column_tk(xlsx_path: Path, sheet: str) -> str:
    import tkinter as tk
    from tkinter import simpledialog

    df = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=2)
    cols = [str(c).strip() for c in df.columns]
    guess = _guess_baseline_column(cols)

    root = tk.Tk()
    root.withdraw()
    try:
        prompt = (
            "Enter the baseline column name containing *all genes* locus tags.\n\n"
            "Columns found:\n" + "\n".join(f"  - {c}" for c in cols)
        )
        chosen = simpledialog.askstring(
            title="decodingpipe",
            prompt=prompt,
            initialvalue=guess,
            parent=root,
        )
    finally:
        root.destroy()

    chosen = (chosen or "").strip()
    return chosen if chosen in cols else guess


def _build_interactive_raw_config() -> Dict:
    """Collect inputs with file pickers and build a minimal config dict."""

    import tkinter as tk
    from tkinter.filedialog import askopenfilename, askdirectory

    root = tk.Tk()
    root.withdraw()

    try:
        mode = _pick_mode_tk(DEFAULT_INPUT_MODE)

        decoding_table = askopenfilename(
            title="Select Codon-tRNAs decoding table (Excel .xlsx)",
            filetypes=[("Excel", "*.xlsx"), ("All files", "*")],
        )
        if not decoding_table:
            raise SystemExit("No Codon-tRNAs decoding table selected. Aborting.")
        decoding_table = Path(decoding_table).expanduser().resolve()
        decoding_table_sheet = _ask_sheet_tk(decoding_table, "Codon-tRNAs decoding table", default="Long Format")

        cds_fasta = askopenfilename(
            title="Select CDS FASTA (.fna/.fa/.fasta)",
            filetypes=[("FASTA", "*.fna *.fa *.fasta *.fas"), ("All files", "*")],
        )
        if not cds_fasta:
            raise SystemExit("No CDS FASTA selected. Aborting.")
        cds_fasta = Path(cds_fasta).expanduser().resolve()

        # Mode-specific inputs
        clusters_xlsx = ""
        clusters_sheet = ""
        genome_allgenes_column = ""
        precomp_rcu = ""
        precomp_devz = ""
        precomp_baseline_sheet = ""

        if mode == "fasta":
            clusters_xlsx = askopenfilename(
                title="Select cluster locus-tags workbook (Excel .xlsx)",
                filetypes=[("Excel", "*.xlsx"), ("All files", "*")],
            )
            if not clusters_xlsx:
                raise SystemExit("No cluster workbook selected. Aborting.")
            clusters_xlsx = str(Path(clusters_xlsx).expanduser().resolve())
            clusters_sheet = _ask_sheet_tk(Path(clusters_xlsx), "Cluster workbook", default="Locus Tags")
            genome_allgenes_column = _ask_baseline_column_tk(Path(clusters_xlsx), clusters_sheet)

        else:
            precomp_rcu = askopenfilename(
                title="Select precomputed per-gene RCU workbook (Excel .xlsx)",
                filetypes=[("Excel", "*.xlsx"), ("All files", "*")],
            )
            if not precomp_rcu:
                raise SystemExit("No precomputed RCU workbook selected. Aborting.")
            precomp_rcu = str(Path(precomp_rcu).expanduser().resolve())

            precomp_devz = askopenfilename(
                title="Select precomputed per-gene devZ (or RCU_devZ) workbook (Excel .xlsx)",
                filetypes=[("Excel", "*.xlsx"), ("All files", "*")],
            )
            if not precomp_devz:
                raise SystemExit("No precomputed devZ workbook selected. Aborting.")
            precomp_devz = str(Path(precomp_devz).expanduser().resolve())

            # Baseline sheet used as "All genes" reference in the precomputed workbooks
            precomp_baseline_sheet = _ask_sheet_tk(Path(precomp_rcu), "Precomputed workbook baseline sheet", default="All genes")

        out_dir = askdirectory(title="Select output directory (Cancel = use ./outputs)")
        out_dir = (Path(out_dir).expanduser().resolve() if out_dir else Path.cwd() / "outputs")

    finally:
        root.destroy()

    raw = {
        "inputs": {
            "mode": mode,
            "decoding_table_xlsx": str(decoding_table),
            "decoding_table_sheet": decoding_table_sheet,
            "cds_fasta": str(cds_fasta),
            # FASTA mode
            "clusters_xlsx": clusters_xlsx,
            "clusters_sheet": clusters_sheet,
            "genome_allgenes_column": genome_allgenes_column,
            "exclude_cluster_columns": [],
            # Precomputed mode
            "precomp_rcu_xlsx": precomp_rcu,
            "precomp_rcudevz_xlsx": precomp_devz,
            "precomp_baseline_sheet": precomp_baseline_sheet,
            "exclude_precomp_sheets": [],
            # do not ask again inside the pipeline
            "use_file_dialog": False,
        },
        "labels": {
            "baseline_label": BASELINE_LABEL,
            "colorbar_label": COLORBAR_LABEL,
        },
        "outputs": {
            "output_dir": str(out_dir),
            "run_tag": "",
            "save_plots": bool(SAVE_PLOTS),
            "show_plots": bool(SHOW_PLOTS),
            "figure_format": FIGURE_FORMAT,
            "figure_dpi": int(FIGURE_DPI),
        },
        "analysis": {
            "output_mode": OUTPUT_MODE,
            "score_agg": SCORE_AGG,
        },
        "plots": {
            "make_trna_shift_heatmap": bool(MAKE_TRNA_SHIFT_HEATMAP),
            "make_mod_load_heatmaps": bool(MAKE_MOD_LOAD_HEATMAPS),
            "make_trna_box_wc_wobble_heatmap": bool(MAKE_TRNA_BOX_WC_WOBBLE_HEATMAP),
            "make_genelevel_wc_wobble_supp": bool(MAKE_GENELEVEL_WC_WOBBLE_SUPP),
            # metric selection per plot: "rcu" or "rcu_devz"
            "trna_metric": "rcu_devz",
            "mod34_metric": "rcu_devz",
            "mod37_metric": "rcu_devz",
            "trna_box_metric": "rcu_devz",
        },

        # Heatmap appearance
        "heatmap_geometry_cm": dict(HEATMAP_GEOMETRY_CM),
        "heatmap_fonts_pt": dict(HEATMAP_FONTS_PT),
        "heatmap_style": dict(HEATMAP_STYLE),

        # Plot-specific options (kept close to USER SETTINGS)
        "plot_trna": {
            "use_manual_trna_order": True,
            "manual_trna_order": list(TRNA_SHIFT_TRNAS_ORDER),
            "show_aa_group_labels": bool(TRNA_SHIFT_SHOW_AA_GROUP_LABELS),
            "draw_group_link_bars": True,
            # Use true curly braces (per group)
            "group_link_style": "bracket", # "brace" or "bracket"
            # Position between row labels (x=0) and AA labels (lab_x)
            "group_link_fraction": 0.75,
            # Brace width (axes units; positive goes to the right)
            "group_link_cap_len_axes": 0.020,
            # Leave a visible gap between adjacent braces (row units)
            "group_link_endpad_rows": 0.18,
        },
        "plot_trna_box": {
            "trnas_to_plot": list(TRNA_BOX_TRNAS_TO_PLOT),
            "show_trna_group_labels": True,
            "draw_group_link_bars": True,
            # Use true curly braces (per group)
            "group_link_style": "bracket", # "brace" or "bracket"
            "group_link_fraction": 0.75,
            "group_link_cap_len_axes": 0.020,
            "group_link_endpad_rows": 0.18,
        },
    }

    return raw


def main():
    raw = _build_interactive_raw_config()
    run_pipeline(raw)


if __name__ == "__main__":
    main()
