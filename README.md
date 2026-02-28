# Bits to Binders: CD20 Protein Binder Design & Analysis

Analysis and datasets for computational protein binder design targeting human CD20. Covers ~12,000 designed sequences from a multi-team competition, including feature engineering, ML analysis, and experimental validation through display assays and SPR.

## Structure

- **`data/`** — Raw and processed datasets (sequences, structural metrics, experimental results)
- **`scripts/`** — Analysis notebooks and Python/R scripts
  - **`adaptyv/`** — Candidate selection and SPR binding affinity analysis
- **`results/`** — Plots and model outputs
- **`misc/`** — Design workflow documentation and kickoff slides

## Key Files

| File | Description |
|------|-------------|
| `data/12k_all_metrics.csv` | Computed features for ML |
| `data/12k_all_results.csv` | Experimental results with enrichment calls (Up/Down/NotSig/Other) |
| `scripts/adaptyv/b2b_summary.csv` | Adaptyv Bio SPR results for selected candidates |


## Citing *Bits to Binders*
