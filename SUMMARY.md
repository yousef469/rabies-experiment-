# Rabies Experiment — Status Summary

> A **learning / experiment project**, not a validated tool and not a metrics
> race. This file is the honest record of what the pipeline does and what it
> does *not* show. The numbers are there to check the reasoning, not to impress.
> Full caveats: [LIMITATIONS.md](LIMITATIONS.md). All metrics: [METRICS.md](METRICS.md).
> Overview + quickstart: [README.md](README.md). Repository: `rabies-experiment-`.

## Goal

Build a reproducible computational pipeline that (a) predicts small-molecule
toxicity with a GNN, (b) screens FDA drugs for repurposing, and (c) models the
PK/PD and viral dynamics of a CNS rabies infection — as a learning/portfolio
project, not as a validated therapeutic claim.

## Environment

- CPU-only (MacBook), Python 3.10, open-source tools.
- Torch 2.2 + `torch_geometric` 2.8, RDKit 2023.09, AutoDock Vina for docking.
- Strictly computational; no wet-lab validation.

## What changed in this audit

The project previously contained several invalid or unreproducible results.
These were corrected:

| Issue | Before | After |
|-------|--------|-------|
| "Cure" definition | `cured = neuron survival > 99%` | cleared = viral + infected-cell extinction, no rebound |
| Headline accuracy | 44.1% ("best validation") | 0.386 held-out test accuracy, macro-F1 0.380, vs 0.288 majority baseline |
| GNN vs DNN | 42.0% vs 39.6% (single random split, no baseline) | scaffold + random splits with majority/stratified baselines |
| Repurposing screen | scored training drugs (leakage) | scores **held-out** test drugs only |
| Docking score | −8.664 kcal/mol (not reproducible) | removed; blind-dock ranking only |
| "50.5% sterilizing cure" | deterministic model never clears | reported as not-applicable / persistence (0/61 parameter combos clear) |
| Hardcoded params | magic numbers scattered | centralised assumed constants in `pkpd_model.py` |

## Pipeline components

| Component | Script | Status | Honest result |
|-----------|--------|--------|---------------|
| PK/PD + viral dynamics | `pkpd_model.py` | Working (assumed params) | Infection persists; no clearance |
| Regional brain model | `spatial_brain.py` | Working (assumed params) | No region clears |
| Parameter sensitivity | `sensitivity_analysis.py` | Working | 0/61 combinations clear |
| Stochastic extinction | `stochastic_cure.py` | Working | ODE never suppresses → SSA not applicable |
| Candidate evaluation | `cure_evaluation.py` | Working | All regimens: not cleared |
| GNN trainer (multi-task) | `train_toxicity.py` | Working | Held-out acc 0.386, macro-F1 0.380 |
| GNN vs DNN benchmark | `benchmark.py` | Working | scaffold: GNN 0.359/0.340 vs DNN 0.339/0.336 |
| Repurposing screen | `screen_repurpose.py` | Working (leakage fixed) | Held-out candidates ranked |
| Docking | `dock_candidates.py`, `vina_dock.py` | Coarse | Blind dock, ranking only |
| GNN explanation | `gnn_explain.py` | Working | Saliency on lead chemotype |
| ADME filter | `adme_filter.py` | Exploratory | Not validated |
| Withdrawn-drug check | `validate_withdrawn.py` | Working | Sanity check only |

## Model performance (honest)

- **GNN (GINConv, 4 layers, hidden=64):** 44,744 params.
  Scaffold split — test acc 0.359, macro-F1 0.340, macro-AUROC 0.607.
  Random split — test acc 0.376, macro-F1 0.366, macro-AUROC 0.641.
- **Fingerprint DNN:** 679,108 params.
  Scaffold split — test acc 0.339, macro-F1 0.336, macro-AUROC 0.596.
- **Baselines:** majority-class macro-F1 0.105 (scaffold) / 0.112 (random).
- **Bottleneck:** dataset size (1,224 compounds). Larger public ToxCast data
  exists but CPU training is slow; this is a genuine, stated limitation, not an
  excuse.

## Docking (blind, coarse)

| Drug | Binding (kcal/mol) |
|------|-------------------:|
| Ripretinib | −8.04 |
| Pralsetinib | −8.03 |
| Selpercatinib | −7.87 |

Blind docking against a surrogate receptor. Ranking only; no validation.

## Blocked / not attempted

- ToxCast pretraining and 5-model ensembling were skipped: they would add only
  modest accuracy for long CPU runs. Not included as results.

## Key files

- [README.md](README.md) — overview and reproduction commands
- [METRICS.md](METRICS.md) — all metrics and baselines
- [LIMITATIONS.md](LIMITATIONS.md) — caveats
- `train_toxicity.py`, `benchmark.py` — ML training + benchmark
- `screen_repurpose.py` — held-out repurposing screen
- `pkpd_model.py`, `spatial_brain.py`, `sensitivity_analysis.py`,
  `cure_evaluation.py`, `stochastic_cure.py` — mechanistic model + UQ
- `data/benchmark_results_{random,scaffold}.json` — benchmark outputs
- `data/repurposing_results.json` — screen output
- `data/sensitivity_results.json` — parameter sweep output
