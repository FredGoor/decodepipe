# decodingpipe: a gene-level tRNA decoding strategy pipeline



This repository contains a Python pipeline to quantify **codon decoding strategy shifts** across gene clusters.

It supports two input modes:

- **FASTA mode**: compute codon counts directly from CDS sequences for each gene list/cluster.
- **Precomputed mode**: start from precomputed per-gene codon-usage metrics (e.g. per-gene RCU and per-gene devZ),
  and reconstruct cluster-level pseudo-count distributions.

The pipeline can generate:
- Cluster codon tables (counts, RCU/RSCU, and annotation columns from the **Codon-tRNAs decoding table**)
- Heatmaps of codon decoding strategies per gene cluster (Watson-Crick vs wobble decoding, tRNA isoacceptor shift, change in tRNA modification dependence) 

## Quick start

1) Create a conda environment:

```bash
conda env create -f environment.yml
conda activate decodingpipe
```

2) Run (interactive, recommended):

```bash
python run_decodingpipe.py
```

Outputs are written to `outputs/` by default (selectable in the file dialog).

## Required inputs

### Codon-tRNAs decoding table (Excel, “true-long” format)

The pipeline expects a sheet containing at least the following columns:

- `AA`
- `Codon 5'-3'`
- `Anticodon 5'-3'`
- `tRNAs`
- `Decoding`
- `Mod34`
- `Mod37`
- `Associated tRMEs`

### Cluster workbook (FASTA mode)

One sheet containing:
- one column listing all genome locus tags (baseline)
- additional columns listing locus tags for each cluster

### CDS FASTA (required)

A FASTA file of CDS sequences where each record header contains a locus tag
(e.g. `locus_tag=...`) or uses the locus tag as the first identifier token.

### Precomputed workbooks (precomputed mode)

Two workbooks, one for per-gene **RCU** and one for per-gene **devZ** (or “RCU_devZ”), with the format:

- Row 1 (from column 2): amino-acid labels
- Row 2 (from column 2): codons (DNA, e.g. `TTT`)
- Rows 4+: genes (column 1) and values (from column 2)

## Plot toggles

For interactive runs, plot toggles (on/off) are defined in the **USER SETTINGS** section at the top of `run_decodingpipe.py`.

## Citation

If you use this code in academic work, please cite the associated manuscript and the archived Zenodo release of this repository.
