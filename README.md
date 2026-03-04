# Data and analysis of CD20-targeting CAR binder domains in the *Bits to Binders* competition

[Link to paper](https://doi.org/10.64898/2026.03.03.709355)

This repository contains the datasets and analyses from the *Bits to Binders* AI-driven protein binder design competition. This repository contains all of the data used to analyze the submitted sequences.

## Structure

- **`data/`** — Raw and processed datasets (sequences, structural metrics, experimental results). The most useful files here are probably "12k_all_metrics.csv" and "12k_all_results.csv".
- **`misc/`** — Extra team methods and kickoff slides
- **`scripts/`** — Analysis notebooks and Python scripts used to perform analysis and generate plots
  - **`adaptyv/`** — Candidate selection for binding affinity study, and SPR results
- **`results/`** — Plots and model outputs

## Key Files

| File | Description |
|------|-------------|
| `data/12k_all_metrics.csv` | Computed features for ML |
| `data/12k_all_results.csv` | Experimental results with enrichment calls (Up/Down/NotSig/Other) |
| `scripts/adaptyv/b2b_summary.csv` | Adaptyv Bio SPR results for selected candidates |


## Citing *Bits to Binders*

(Journal and DOI coming soon)

Kosonocky, C.W., Abel, A.M., Feller, A.L., Cifuentes Rieffer, A.E., Woolley, P.R., Lála, J., Barth, D.R., Gardner, T., Bits to Binders Competitors, Ekker, S.C., Ellington, A.D., Wierson, W.A., & Marcotte, E.M. (2026). Validation and analysis of 12,000 AI-driven CAR-T designs in the Bits to Binders competition. bioRxiv. https://doi.org/10.64898/2026.03.03.709355

```bibtex
@article{kosonocky2026bits,
  title={Validation and analysis of 12,000 AI-driven CAR-T designs in the Bits to Binders competition},
  author={Kosonocky, Clayton W. and Abel, Alex M. and Feller, Aaron L. and Cifuentes Rieffer, Amanda E. and Woolley, Phillip R. and L{\'a}l{\'a}, Jakub and Barth, Daryl R. and Gardner, Tynan and {Bits to Binders Competitors} and Ekker, Stephen C. and Ellington, Andrew D. and Wierson, Wesley A. and Marcotte, Edward M.},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.03.03.709355}
}
```
