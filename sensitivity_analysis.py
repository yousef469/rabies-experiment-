"""
sensitivity_analysis.py — How much does the "cure" verdict depend on assumed parameters?

Why this exists
---------------
The PK/PD and viral-dynamics models in this project run on ASSUMED parameters
(EC50, Hill coefficient, viral kill rate, infection rate, treatment delay, dose).
None of them were fitted to rabies data. Any single "X% cure" number therefore
implies a precision the model does not have.

This script quantifies that: it re-runs the model while sweeping one assumption at
a time and reports how the outcome classification (CLEARED / SUPPRESSED / ONGOING)
and the peak viral load move. If a conclusion flips inside a plausible parameter
range, the conclusion is not robust and must be stated as such.

Outputs
-------
* A human-readable table to stdout.
* data/sensitivity_results.json with the raw sweep values.
"""
import json
import os

import numpy as np

import pkpd_model as M

DATA_DIR = 'data'
LEAD_SMILES = 'CC1=CC(=O)NC2=CC=CC=C12'
LEAD_DOSE_MG = 500
TREATMENT_DELAY_HRS = 48


def _outcome_label(viral):
    if viral.get('cleared'):
        return 'CLEARED'
    if viral.get('suppressed'):
        return 'SUPPRESSED'
    return 'ONGOING'


def run_once(smiles=LEAD_SMILES, dose_mg=LEAD_DOSE_MG, delay_h=TREATMENT_DELAY_HRS):
    """Run one PK/PD + viral-dynamics evaluation and return a compact dict."""
    pkpd = M.simulate_pkpd(smiles, dose_mg=dose_mg, name='sensitivity')
    viral = M.simulate_viral_dynamics(pkpd, treatment_delay_hrs=delay_h)
    return {
        'peak_virus': float(viral['peak_virus']),
        'final_virus': float(viral['final_virus']),
        'final_infected': float(viral['final_infected']),
        'neurons_preserved': float(viral['survival_fraction']),
        'outcome': _outcome_label(viral),
    }


def sweep(name, attr, values, base, **run_kwargs):
    """Sweep a single module-level constant and restore it afterwards."""
    original = getattr(M, attr)
    rows = []
    for v in values:
        setattr(M, attr, v)
        res = run_once(**run_kwargs)
        res.update({'param': name, 'value': v})
        rows.append(res)
    setattr(M, attr, original)
    return rows


def grid(attr_x, values_x, attr_y, values_y, **run_kwargs):
    """2-D sweep of two module-level constants."""
    ox, oy = getattr(M, attr_x), getattr(M, attr_y)
    rows = []
    for vx in values_x:
        setattr(M, attr_x, vx)
        for vy in values_y:
            setattr(M, attr_y, vy)
            res = run_once(**run_kwargs)
            res.update({attr_x: vx, attr_y: vy})
            rows.append(res)
    setattr(M, attr_x, ox)
    setattr(M, attr_y, oy)
    return rows


def _print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'param':<18}{'value':>10}{'peak V':>12}{'final V':>12}{'neurons%':>10}{'outcome':>12}")
    print('  ' + '-' * 72)
    for r in rows:
        print(f"  {r['param']:<18}{r['value']:>10g}{r['peak_virus']:>12.4g}"
              f"{r['final_virus']:>12.4g}{r['neurons_preserved']*100:>9.1f}%{r['outcome']:>12}")


def main():
    base = run_once()
    print('=' * 76)
    print('SENSITIVITY ANALYSIS — does the "cure" verdict survive the assumptions?')
    print('=' * 76)
    print(f"Lead SMILES: {LEAD_SMILES}   dose={LEAD_DOSE_MG} mg   delay={TREATMENT_DELAY_HRS} h")
    print(f"Baseline outcome: {base['outcome']}   peak V={base['peak_virus']:.4g}   "
          f"final V={base['final_virus']:.4g}")

    all_rows = []

    sweeps = [
        ('PD EC50 (uM)',        'PD_EC50_UM',    [1, 3, 10, 30, 100]),
        ('Viral EC50 (uM)',     'VIRAL_EC50_UM', [1, 3, 6, 12, 24]),
        ('Viral kill alpha',    'VIRAL_ALPHA',   [0.05, 0.15, 0.25, 0.5, 1.0]),
        ('Infection beta',      'VIRAL_BETA',    [0.003, 0.006, 0.012, 0.024, 0.048]),
        ('Inf. death delta',    'VIRAL_DELTA',   [0.005, 0.01, 0.02, 0.04, 0.08]),
    ]
    for title, attr, values in sweeps:
        rows = sweep(title, attr, values, base)
        _print_table(f"[1-D] {title}  (baseline = {getattr(M, attr)})", rows)
        all_rows.extend(rows)

    # Treatment timing and dose are clinical knobs, not model constants.
    delay_rows = []
    for d in [24, 48, 72, 96, 120, 168]:
        res = run_once(delay_h=d)
        res.update({'param': 'treatment delay (h)', 'value': d})
        delay_rows.append(res)
    _print_table('[1-D] treatment delay', delay_rows)
    all_rows.extend(delay_rows)

    dose_rows = []
    for dose in [50, 100, 250, 500, 1000]:
        res = run_once(dose_mg=dose)
        res.update({'param': 'dose (mg)', 'value': dose})
        dose_rows.append(res)
    _print_table('[1-D] dose', dose_rows)
    all_rows.extend(dose_rows)

    # 2-D: the two EC50s are the least-justified assumptions, so cross them.
    g_rows = grid('PD_EC50_UM', [1, 3, 10, 30, 100],
                  'VIRAL_EC50_UM', [1, 3, 6, 12, 24])
    print('\n[2-D] outcome grid: rows = PD EC50 (uM), cols = viral EC50 (uM)')
    header = '  ' + ' ' * 8 + ''.join(f'{v:>10}' for v in [1, 3, 6, 12, 24])
    print(header)
    for px in [1, 3, 10, 30, 100]:
        cells = [r for r in g_rows if r['PD_EC50_UM'] == px]
        cells.sort(key=lambda r: r['VIRAL_EC50_UM'])
        line = f"  {px:>6}  " + ''.join(f"{r['outcome'][:8]:>10}" for r in cells)
        print(line)
    all_rows.extend(g_rows)

    outcomes = [r['outcome'] for r in all_rows]
    n_clear = outcomes.count('CLEARED')
    print('\n' + '=' * 76)
    print(f"Across {len(outcomes)} parameter combinations: "
          f"CLEARED in {n_clear} ({100*n_clear/len(outcomes):.0f}%), "
          f"SUPPRESSED in {outcomes.count('SUPPRESSED')}, "
          f"ONGOING in {outcomes.count('ONGOING')}.")
    print('Interpretation: the verdict is assumption-dependent. Quote a RANGE, not a point value.')
    print('=' * 76)

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, 'sensitivity_results.json')
    with open(out, 'w') as f:
        json.dump({'baseline': base, 'rows': all_rows}, f, indent=2)
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
