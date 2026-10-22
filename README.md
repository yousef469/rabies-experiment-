# Rabies Experiment (rabies-experiment-)

A **learning / experiment project**, not a validated tool and not a metrics race.

This started as an attempt to build a computational drug-discovery pipeline for
rabies. The most valuable thing it produced was not a result — it was the
discovery that my own first results were wrong, and the process of finding out
why. The project is framed as an experiment: a question, a pipeline, a rigorous
check, and an honest conclusion.

> **What this is:** an experiment in computational drug discovery, and a record
> of learning how to evaluate models honestly.
> **What this is not:** a cure, a validated drug, or a benchmark record. The
> numbers exist to *check my reasoning*, not to impress. See
> [LIMITATIONS.md](LIMITATIONS.md).

---

## The question

Can a small computational pipeline — a graph neural network for toxicity, plus
a mechanistic model of rabies infection — nominate a credible drug candidate?

**Honest answer after the experiment: no, not a cure.** It can nominate
*candidates to test*, and it can tell us clearly what it *cannot* tell us.

---

## What we have, and what each piece does

### Machine-learning half

| Component | File | What it does |
|-----------|------|--------------|
| Multi-task toxicity GNN | `train_toxicity.py` | Trains a GINConv graph network on 4 datasets (DILIrank, ClinTox, Tox21, SIDER) with one shared backbone and several prediction heads. |
| Fair benchmark | `benchmark.py` | Compares the GNN against a fingerprint DNN using scaffold *and* random splits, with majority-class and stratified baselines. This is the part that checks whether the model is actually useful. |
| Repurposing screen | `screen_repurpose.py` | Screens FDA drugs — but **only the GNN's held-out test drugs** — for low predicted toxicity, then optionally docks them. |
| GNN explainer | `gnn_explain.py` | Gradient saliency: which atoms drive a toxicity prediction. |
| Withdrawn-drug check | `validate_withdrawn.py` | Sanity check on drugs pulled from the market. |

### Mechanistic (physics/biology) half

| Component | File | What it does |
|-----------|------|--------------|
| PK/PD + viral dynamics | `pkpd_model.py` | Two-compartment pharmacokinetics, a Hill dose-response, and ODEs for infected cells, virus, and immune response. |
| Spatial brain model | `spatial_brain.py` | Splits the brain into regions and spreads the infection between them. |
| Uncertainty sweep | `sensitivity_analysis.py` | Re-runs the model across many assumed parameter values to see if the conclusion survives. |
| Candidate evaluation | `cure_evaluation.py` | End-to-end run for the selected lead: PK, viral outcome, combination, spatial. |
| Stochastic extinction | `stochastic_cure.py` | ODE first, then a Gillespie random simulation *if* the infection is suppressed. |

### Molecule design + docking

| Component | File | What it does |
|-----------|------|--------------|
| Genetic algorithm | `evolve_molecules.py` | Mutates/crosses molecules toward a heuristic fitness score. |
| Molecule generation | `generate_molecules.py` | Builds the starting library. |
| Docking | `vina_dock.py`, `dock_candidates.py` | AutoDock Vina against the rabies glycoprotein (PDB 6LGX). Coarse, blind docking. |

---

## How the experiment actually went

This is the real content of the project — the sequence of finding out I was wrong:

1. **Built the pipeline.** Toxicity GNN + repurposing screen + viral dynamics model.
2. **Reported a "cure."** The model said ~99% of neurons survived, so I called it a cure.
3. **Audited the numbers.** The accuracy figure was "best validation accuracy," not test
   performance. The screen was scoring drugs the model had trained on. The docking score
   didn't reproduce.
4. **Found the root cause of the "cure."** I had defined *cured* as ">99% of neurons
   survived." But the virus and infected cells were still there. Survival is not clearance.
5. **Fixed the definition and the evaluation.** Corrected `cured` to mean viral
   extinction with no rebound; rebuilt the benchmark with scaffold splits and baselines;
   removed the screen's leakage; added a parameter sweep.
6. **Reported the honest result:** under every assumed parameter combination tested,
   **the model does not clear the infection (0 of 61)**.

The corrected framing is in [SUMMARY.md](SUMMARY.md); the numbers are in
[METRICS.md](METRICS.md); the caveats are in [LIMITATIONS.md](LIMITATIONS.md).

---

## What this project is evidence for

Not "I found a rabies drug." Instead:

- I can build an end-to-end computational pipeline (ML + chemistry + ODEs).
- I understand **data leakage**, **scaffold splits**, and **baselines** — and I
  actively design against them.
- I can tell the difference between a number and *evidence*, and between
  *survival* and *clearance*.
- I corrected my own headline result when the evidence didn't support it.

That last point is the point of the project.

---

## Results (supporting evidence, not the goal)

| Quantity | Value | Honest reading |
|----------|------:|----------------|
| DILIrank drugs | 1,224 | — |
| GNN params / DNN params | 44,744 / 679,108 | GNN is ~15× smaller |
| GNN test macro-F1 (scaffold split) | 0.340 | modest |
| Majority-class baseline macro-F1 | 0.105 | the bar the model must beat |
| Parameter combinations that clear the infection | 0 / 61 | infection persists |

Full tables: [METRICS.md](METRICS.md).

---

## Reproduce

Python 3.10.

```bash
pip install -r requirements.txt

python benchmark.py --split both        # fair GNN-vs-DNN benchmark
python screen_repurpose.py              # held-out repurposing screen
python cure_evaluation.py               # end-to-end candidate evaluation
python sensitivity_analysis.py          # parameter uncertainty sweep
python stochastic_cure.py               # stochastic extinction analysis
```

Docking needs the `vina` binary and the cached receptor in `pdb_cache/`.

---

## Repository notes

- No model weights or large datasets are committed (see `.gitignore`); download
  them / re-run the scripts to regenerate.
- Some scripts are exploratory and not part of the validated pipeline; their
  status is listed in [SUMMARY.md](SUMMARY.md).
- This is strictly computational. There is no wet-lab work.
