# METRICS.md

> This is a **learning / experiment project**. The numbers below are supporting
> evidence used to check the reasoning — they are not the goal of the project,
> and a modest number reported honestly is the intended outcome. See
> [LIMITATIONS.md](LIMITATIONS.md) for what these numbers do *not* mean.

All numbers below are reproduced from the scripts in this repository. Raw JSON
artifacts are written to `data/`.

---

## 1. Toxicity GNN — DILIrank 4-class classification

Dataset: DILIrank 2.0 (FDA), 1,224 drugs with valid SMILES and a DILI-concern
label. Classes: Ambiguous / No-concern / Less-concern / Most-concern.

### 1.1 `benchmark.py` — GNN (GINConv) vs DNN (molecular fingerprints)

Split sizes are identical for both splits: 734 train / 245 val / 245 test.
Models are selected on validation and reported on the held-out test split.

**Scaffold split (Murcko scaffolds; the honest number).** The train set contains
811 distinct scaffolds, so test scaffolds are largely unseen.

| Model | Test acc | Macro-F1 | Weighted-F1 | Macro-AUROC | Params |
|-------|---------:|---------:|------------:|------------:|-------:|
| Majority class | 0.265 | 0.105 | 0.111 | — | — |
| Stratified random | 0.249 | 0.246 | 0.248 | — | — |
| DNN (fingerprints) | 0.339 | 0.336 | 0.334 | 0.596 | 679,108 |
| GNN (GINConv) | **0.359** | **0.340** | 0.347 | **0.607** | 44,744 |

**Random split (stratified; optimistic upper bound).**

| Model | Test acc | Macro-F1 | Weighted-F1 | Macro-AUROC | Params |
|-------|---------:|---------:|------------:|------------:|-------:|
| Majority class | 0.290 | 0.112 | 0.130 | — | — |
| Stratified random | 0.298 | 0.279 | 0.296 | — | — |
| DNN (fingerprints) | 0.371 | 0.351 | 0.359 | 0.600 | 679,108 |
| GNN (GINConv) | **0.376** | **0.366** | 0.369 | **0.641** | 44,744 |

Run time (CPU): scaffold — DNN 7.7 s, GNN 29.9 s; random — DNN 16.2 s, GNN 47.5 s.

### 1.2 `train_toxicity.py` — multi-task model, held-out TEST split

Trained jointly on DILIrank + ClinTox + Tox21 + SIDER. Split: 856 / 184 / 184.

| Metric (DILIrank test) | Value |
|------------------------|------:|
| Best validation accuracy (model selection only) | 0.440 |
| Majority-class baseline accuracy | 0.288 |
| Majority-class baseline macro-F1 | 0.112 |
| **GNN held-out test accuracy** | **0.386** |
| **GNN held-out test macro-F1** | **0.380** |

> The 0.440 figure is **validation accuracy used for early stopping**, not a
> test result. It must not be quoted as model performance.

**Interpretation.** The GNN is only a few points above the majority-class and
stratified-random baselines, and its advantage over the fingerprint DNN is small
(+0.02 accuracy, +0.004 macro-F1 on the scaffold split). It is *not* a
production-grade toxicity predictor. Macro-F1 in the 0.34–0.38 range is the
honest summary.

---

## 2. PK/PD and viral-dynamics model (`pkpd_model.py`, `spatial_brain.py`)

All parameters are assumed (see [LIMITATIONS.md](LIMITATIONS.md)). The model
outputs below are **model predictions**, not experimental results.

### 2.1 Baseline viral trajectory (assumed parameters)

| Quantity | Value |
|----------|------:|
| Peak virus (relative units) | 3,259 |
| Final virus | 3,259 |
| Final infected cells | 920 |
| Fraction of neurons "preserved" | 0.998 |
| **Outcome** | **ONGOING (not cleared)** |

Note the separation between *neuron survival* and *viral clearance*: the model
can report ~99.8% neuron preservation while the infection is still active. These
are different things and the old code conflated them.

### 2.2 `sensitivity_analysis.py` — 61 parameter combinations

Swept: PD EC50, viral EC50, kill rate α, infection rate β, death rate δ,
treatment delay (24–168 h), dose (50–1000 mg), plus a 2-D PD-EC50 × viral-EC50
grid.

| Outcome | Count | Fraction |
|---------|------:|---------:|
| Cleared | 0 | 0.0% |
| Suppressed | 2 | 3.3% |
| Ongoing | 59 | 96.7% |

**No combination of the assumed parameters produced viral clearance.**

### 2.3 `cure_evaluation.py` — evaluation of the selected lead

| Regimen | Neuron survival | Cleared? |
|---------|----------------:|---------:|
| IV | 97.3% | No |
| Oral (F = 0.70) | 95.7% | No |
| Oral (F = 0.50) | 93.7% | No |
| Oral (F = 0.30) | 89.2% | No |
| Illustrative 2-drug combination | 99.4% | No |

### 2.4 `stochastic_cure.py` — hybrid ODE + Gillespie

The deterministic phase never suppresses the infection (final V ≈ 125,
I ≈ 47), so the stochastic-extinction phase is **not applicable** and the script
reports this instead of a cure.

---

## 3. Docking (blind, coarse)

AutoDock Vina, box centred on the receptor centroid, 26 Å cube,
exhaustiveness 8. These are ranking scores only — no binding-mode validation,
no redocking control, no replicates.

| Drug | Vina binding energy (kcal/mol) |
|------|-------------------------------:|
| Ripretinib | −8.04 |
| Pralsetinib | −8.03 |
| Selpercatinib | −7.87 |

> Earlier drafts quoted a single −8.664 kcal/mol figure. That number is **not
> reproducible** and has been removed.

---

## 4. What is *not* claimed

- No SOTA-beating claim. Literature DILI models reach higher scores on larger or
  proprietary datasets; this model is not comparable to them.
- No cure. The mechanistic model predicts persistence under every parameter
  combination tested.
- No validated binding. Docking here is a coarse filter.
- No ADME prediction validated against experiment.
