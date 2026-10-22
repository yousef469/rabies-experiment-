# LIMITATIONS.md

This file lists the scientific and technical caveats of the project. It is
intentionally blunt. If a claim is not supported, it should not be made.

## 1. The PK/PD and viral models are not calibrated

- Every pharmacokinetic, pharmacodynamic, and viral parameter in `pkpd_model.py`,
  `spatial_brain.py`, `cure_evaluation.py`, and `stochastic_cure.py` is an
  **assumed placeholder**. None was fitted to rabies data, human CNS
  pharmacokinetics, or RABV replication kinetics.
- Rabies-specific quantities (neuron infection rate β, infected-cell death rate
  δ, virion production p, immune parameters) are chosen for plausible orders of
  magnitude, not measured values.
- Consequence: the model can be tuned to produce almost any outcome. Therefore
  the *absolute* predictions (e.g. "97% neuron survival") carry no evidential
  weight. Only the *sensitivity trends* — how outcomes move as parameters move —
  are informative, and even those are conditional on the model structure.

## 2. "Cleared" vs. "survived" — a definition that was previously wrong

- The original code labelled an outcome "cured" when >99% of neurons survived.
  That is not clearance: an infection can persist indefinitely while most
  neurons remain alive. Under the corrected definition (virions and infected
  cells driven to extinction with no rebound), the model predicts **persistence,
  not clearance** in 59/61 parameter combinations.
- This is the single most important correction in the project. Prior "cure"
  numbers were an artifact of the definition, not a discovery.

## 3. The toxicity model is weak

- Held-out macro-F1 is ≈0.34–0.38; macro-AUROC ≈0.60–0.64. The model is only
  slightly above majority-class and stratified-random baselines.
- DILIrank is small (1,224 compounds) and imbalanced. Random splits overestimate
  performance because near-duplicate scaffolds leak across the split; the
  scaffold split is the honest number and is reported as the primary result.
- The GNN's advantage over a fingerprint DNN is small (~+0.02 accuracy on the
  scaffold split). The main justification for a GNN here is parameter
  efficiency, not a large accuracy gain.

## 4. No data leakage in the repurposing screen — and why it matters

- `screen_repurpose.py` previously selected "low-toxicity" drugs by running the
  GNN on the same DILIrank drugs it was trained on. That is circular: the model
  is asked to confirm its own training labels.
- The script now reproduces the trainer's split (seed 42) and screens **only the
  held-out test drugs**. If the split cannot be reproduced it stops rather than
  silently scoring training compounds.

## 5. Docking is a coarse filter, not a binding prediction

- `screen_repurpose.py --dock`, `dock_candidates.py`, and `vina_dock.py` run
  **blind docking** (box centred on the receptor centroid). Blind docking does
  not indicate a real binding site.
- There is no redocking control, no replicate seeds in the default path, no
  measured affinity, and no experimental validation.
- The previously quoted −8.664 kcal/mol is **not reproducible** and has been
  removed.

## 6. Target and receptor choices are unvalidated

- Rabies has no well-established small-molecule drug target. The receptor used
  for docking (e.g. PDB 6LGX) was chosen as a plausible surrogate, not because
  it is a validated anti-rabies target.
- The "lead molecule" is an arbitrary output of the design pipeline; it is not
  the result of a validated structure-activity study.

## 7. Metrics and reporting

- Accuracy alone is misleading on this imbalanced 4-class problem. All reported
  results now use macro-F1 and macro-AUROC and are shown next to explicit
  baselines.
- Validation accuracy (used for early stopping) is clearly labelled and must not
  be reported as test performance.
- Some notebooks/scripts in the repository are exploratory and are *not* part of
  the validated pipeline; see [SUMMARY.md](SUMMARY.md) for per-file status.

## 8. Scope

- This is a learning/portfolio project. It is strictly computational. There is
  no wet-lab component, no cell culture, no animal work, and no clinical data.
- Nothing here should be interpreted as medical advice or as evidence that any
  compound treats rabies.
