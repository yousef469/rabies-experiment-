"""
Candidate Evaluation:

1. Toxicity / side-effect flags of the lead molecule
2. Oral vs IV pharmacokinetics
3. Illustrative two-drug combination (Bliss independence)
4. Parameter sensitivity of the spatial brain model

IMPORTANT: every number printed here is a MODEL OUTPUT under assumed parameters.
None of the PK/PD or viral parameters were fitted to rabies data. "Cleared" means
the simulated virus and infected-cell populations were driven to extinction with
no rebound — it is NOT a clinical claim. See LIMITATIONS.md.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, '.')

from rdkit import Chem
from rdkit.Chem import Descriptors
from pkpd_model import (
    simulate_pkpd_multi_dose, simulate_pkpd_infusion, simulate_viral_dynamics_immune,
    predict_toxicity, get_bbb_permeability, PD_EC50_UM, PD_HILL,
    VIRAL_EC50_UM, VIRAL_ALPHA,
)
from spatial_brain import simulate_spatial_rabies, print_report

# Single source of truth for the lead molecule: whatever the GA/docking pipeline
# actually selected. Falls back to the known lead if the artifact is missing.
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(_HERE, 'cure_found.json')) as f:
        LEAD_SMILES = json.load(f)['smiles']
except (OSError, KeyError, json.JSONDecodeError):
    LEAD_SMILES = 'CC1=CC(=O)NC2=CC=CC=C12'

print("=" * 70)
print("  CANDIDATE EVALUATION (model output under assumed parameters)")
print("  SMILES: " + LEAD_SMILES)
print("=" * 70)

mol = Chem.MolFromSmiles(LEAD_SMILES)
mw = Descriptors.MolWt(mol)
daily_dose = 5e-3 * mw * 1000  # 5 mmol daily

# ============================================================
# 1. TOXICITY / SIDE EFFECTS
# ============================================================
print("\n" + "=" * 70)
print("  [1/4] TOXICITY & SIDE EFFECTS")
print("=" * 70)

tox = predict_toxicity(LEAD_SMILES)
print(f"  Overall toxicity score: {tox.get('score', 'N/A')}")
print(f"  Toxic: {tox.get('toxic', 'N/A')}")
print(f"  Flags: {tox.get('flags', [])}")

# Detailed molecular property analysis
from rdkit.Chem import Descriptors, Crippen, Lipinski
print(f"\n  --- Molecular Properties ---")
print(f"  MW:           {mw:.0f} g/mol")
print(f"  LogP:         {Crippen.MolLogP(mol):.2f}")
print(f"  TPSA:         {Descriptors.TPSA(mol):.1f} A^2")
print(f"  HBD:          {Lipinski.NumHDonors(mol)}")
print(f"  HBA:          {Lipinski.NumHAcceptors(mol)}")
print(f"  RotBonds:     {Descriptors.NumRotatableBonds(mol)}")
print(f"  AromRings:    {Descriptors.NumAromaticRings(mol)}")
print(f"  Lipinski:     {'PASS' if Lipinski.NumHDonors(mol) <= 5 and Lipinski.NumHAcceptors(mol) <= 10 and Crippen.MolLogP(mol) <= 5 and mw <= 500 else 'FAIL'}")
print(f"  BBB score:    {get_bbb_permeability(LEAD_SMILES)*100:.1f}%")

# ============================================================
# 2. ORAL BIOAVAILABILITY
# ============================================================
print("\n" + "=" * 70)
print("  [2/4] ORAL BIOAVAILABILITY VS IV")
print("=" * 70)

def simulate_oral(smiles, dose_mg, bioavailability=0.5, n_doses=14, interval_hrs=24):
    """Simulate oral dosing with bioavailability fraction."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import numpy as np
    from scipy.integrate import solve_ivp
    from pkpd_model import get_bbb_permeability, PARAMS, pk_iv

    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol) if mol else 300
    pBBB = get_bbb_permeability(smiles)
    params = PARAMS.copy()
    params['pBBB'] = pBBB
    params['MW'] = mw

    total_hrs = interval_hrs * n_doses + 48
    dt = 0.5
    t_eval = np.arange(0, total_hrs + dt, dt)
    A_plasma = 0.0
    A_brain = 0.0
    all_Cp, all_Cb = [], []

    dose_mmol = dose_mg / mw * bioavailability  # Only fraction reaches plasma

    for dose_idx in range(n_doses):
        t_start = dose_idx * interval_hrs
        t_end = min((dose_idx + 1) * interval_hrs, total_hrs)
        t_span = (t_start, t_end)
        t_seg = np.arange(t_start, t_end + dt, dt)
        A_plasma += dose_mmol
        sol = solve_ivp(pk_iv, t_span, [A_plasma, A_brain],
                        args=(params,), t_eval=t_seg, method='RK45', rtol=1e-4, atol=1e-6)
        A_plasma = sol.y[0, -1]
        A_brain = sol.y[1, -1]
        Cp_seg = sol.y[0] / params['V_plasma'] * 1000
        Cb_seg = sol.y[1] / params['V_brain'] * 1000
        if dose_idx == 0:
            all_Cp, all_Cb = list(Cp_seg), list(Cb_seg)
        else:
            all_Cp.extend(list(Cp_seg[1:]))
            all_Cb.extend(list(Cb_seg[1:]))

    all_Cp, all_Cb = np.array(all_Cp), np.array(all_Cb)
    t_out = np.linspace(0, len(all_Cb) * dt, len(all_Cb))
    EC50, hill = PD_EC50_UM, PD_HILL
    suppression = all_Cb**hill / (EC50**hill + all_Cb**hill)
    return {
        't': t_out,
        'C_plasma_uM': all_Cp, 'C_brain_uM': all_Cb,
        'Cmax_brain_uM': all_Cb.max(), 'Cmin_brain_uM': all_Cb.min(),
        'trough_brain_uM': all_Cb[-int(24/dt):].min() if len(all_Cb) > int(24/dt) else all_Cb.min(),
        'avg_suppression': suppression.mean(),
        'time_above_ec50_hrs': np.sum(all_Cb >= EC50) * dt,
        'AUC_brain': np.trapz(all_Cb, t_out),
        'n_doses': n_doses, 'interval_hrs': interval_hrs,
    }

print(f"  --- IV vs Oral (same 5 mmol daily dose) ---")
print(f"  {'Regimen':<25} {'Cmax_b':>7} {'Cmin_b':>7} {'Suppr':>6} {'Surv':>6} {'Cleared':>8}")
print(f"  {'-'*59}")

for bioavail, label in [(1.0, 'IV (100% bioavail)'), (0.7, 'Oral F=0.70'), (0.5, 'Oral F=0.50'), (0.3, 'Oral F=0.30')]:
    if bioavail == 1.0:
        pkpd = simulate_pkpd_multi_dose(LEAD_SMILES, dose_mg=daily_dose, n_doses=14, interval_hrs=24)
    else:
        pkpd = simulate_oral(LEAD_SMILES, dose_mg=daily_dose, bioavailability=bioavail, n_doses=14, interval_hrs=24)

    viral = simulate_viral_dynamics_immune(pkpd, duration_hrs=720)
    cmax = pkpd.get('Cmax_brain_uM', 0)
    cmin = pkpd.get('trough_brain_uM', pkpd.get('Cmin_brain_uM', 0))
    supp = pkpd.get('avg_suppression', 0) * 100
    surv = viral['survival_fraction'] * 100
    cleared = 'YES' if viral.get('cleared') else 'NO'
    print(f"  {label:<25} {cmax:>6.1f} {cmin:>6.2f} {supp:>5.1f}% {surv:>5.1f}% {cleared:>8}")

# ============================================================
# 3. MULTI-DRUG THERAPY (ILLUSTRATIVE)
# ============================================================
print("\n" + "=" * 70)
print("  [3/4] MULTI-DRUG COMBINATION (illustrative additive model)")
print("=" * 70)
print("  Combination effect = Bliss independence of the two single-agent")
print("  dose-response curves. Favipiravir's assumed EC50=0.5 uM is a literature-")
print("  flavoured guess, NOT a fitted rabies value.")

# Multi-drug scenarios
fav_smi = 'CC1=NC(=C2N1C=C(C(=N2)C(=O)O)C(=O)N)C(=O)O'  # Favipiravir (T705)
FAV_EC50_UM = 0.5        # ASSUMED favipiravir brain EC50 (uM) - not fitted to rabies

print(f"  {'Therapy':<35} {'Cmax_b':>7} {'Cmin_b':>7} {'Suppr':>6} {'Surv':>6} {'Cleared':>8}")
print(f"  {'-'*71}")

def eval_therapy(smiles, dose_mg, n_doses=14, interval_hrs=24):
    pkpd = simulate_pkpd_multi_dose(smiles, dose_mg=dose_mg, n_doses=n_doses, interval_hrs=interval_hrs)
    viral = simulate_viral_dynamics_immune(pkpd, duration_hrs=720)
    return pkpd, viral


def effective_conc(target_supp, ec50=VIRAL_EC50_UM, hill=1.5, cap=0.99):
    """Invert the viral model's Hill dose-response.

    The viral ODE applies drug_eff = alpha * D^h / (EC50_v^h + D^h). Given a target
    fractional suppression s (here the Bliss-combined suppression) this returns the
    concentration D that reproduces it, so the combination can be fed through the
    same single-drug solver self-consistently.
    """
    s = np.clip(np.asarray(target_supp, dtype=float), 0.0, cap)
    return ec50 * (s / (1.0 - s)) ** (1.0 / hill)


# 1. Lead alone QD
pkpd_qd, viral_qd = eval_therapy(LEAD_SMILES, daily_dose, 14, 24)
print(f"  {'Lead alone (IV QD)':<35} {pkpd_qd['Cmax_brain_uM']:>6.1f} {pkpd_qd['trough_brain_uM']:>6.2f} "
      f"{pkpd_qd['avg_suppression']*100:>5.1f}% {viral_qd['survival_fraction']*100:>5.1f}% "
      f"{'YES' if viral_qd.get('cleared') else 'NO':>8}")

# 2. Lead alone BID
pkpd_bid, viral_bid = eval_therapy(LEAD_SMILES, daily_dose/2, 28, 12)
print(f"  {'Lead alone (IV BID)':<35} {pkpd_bid['Cmax_brain_uM']:>6.1f} {pkpd_bid['trough_brain_uM']:>6.2f} "
      f"{pkpd_bid['avg_suppression']*100:>5.1f}% {viral_bid['survival_fraction']*100:>5.1f}% "
      f"{'YES' if viral_bid.get('cleared') else 'NO':>8}")

# 3. Favipiravir alone BID
pkpd_fav, viral_fav = eval_therapy(fav_smi, daily_dose/2, 28, 12)
print(f"  {'Favipiravir alone (IV BID)':<35} {pkpd_fav['Cmax_brain_uM']:>6.1f} {pkpd_fav['trough_brain_uM']:>6.2f} "
      f"{pkpd_fav['avg_suppression']*100:>5.1f}% {viral_fav['survival_fraction']*100:>5.1f}% "
      f"{'YES' if viral_fav.get('cleared') else 'NO':>8}")

# 4. Combination via Bliss independence. Each agent's own suppression curve is
#    computed with its own EC50, combined by Bliss, then converted back to a single
#    effective concentration series fed to the SAME viral solver (no discarding).
pkpd_a, _ = eval_therapy(LEAD_SMILES, daily_dose/2, 28, 12)
pkpd_b, _ = eval_therapy(fav_smi, daily_dose/2, 28, 12)

t_min = min(len(pkpd_a['C_brain_uM']), len(pkpd_b['C_brain_uM']))
Ca = pkpd_a['C_brain_uM'][:t_min]
Cb = pkpd_b['C_brain_uM'][:t_min]
t_out = np.linspace(0, 28 * 12, t_min)

supp_a = Ca**PD_HILL / (PD_EC50_UM**PD_HILL + Ca**PD_HILL)
supp_b = Cb**PD_HILL / (FAV_EC50_UM**PD_HILL + Cb**PD_HILL)
supp_combo = np.minimum(1 - (1 - supp_a) * (1 - supp_b), 1.0)  # Bliss independence

combo_dr = {'t': t_out, 'C_brain_uM': effective_conc(supp_combo)}
viral_combo = simulate_viral_dynamics_immune(combo_dr, duration_hrs=720)
print(f"  {'Lead BID + Favipiravir BID (Bliss)':<35} {max(Ca.max(), Cb.max()):>6.1f} {'--':>6} "
      f"{supp_combo.mean()*100:>5.1f}% {viral_combo['survival_fraction']*100:>5.1f}% "
      f"{'YES' if viral_combo.get('cleared') else 'NO':>8}")
print("  Note: combination uses the Bliss-combined suppression fed back through the")
print("  single-drug solver, so the 'Surv' column reflects that additive effect.")

# ============================================================
# 4. SENSITIVITY ANALYSIS (Spatial Model)
# ============================================================
print("\n" + "=" * 70)
print("  [4/4] SPATIAL MODEL SENSITIVITY ANALYSIS")
print("=" * 70)
print(f"  Varying delay/dose/interval to see whether clearance is ever reached...")
print(f"  {'Scenario':<35} {'Overall':>8} {'Brainstem':>9} {'Cortex':>7} {'Cleared':>8}")
print(f"  {'-'*69}")

def run_spatial(cmax, trough, delay, interval, label):
    r = simulate_spatial_rabies(cmax, trough, delay, interval, total_hrs=720)
    g = r['_global']
    bs_surv = r.get('Brainstem', {}).get('survival_fraction', 0) * 100
    cx_surv = r.get('Cortex', {}).get('survival_fraction', 0) * 100
    ov_surv = g['overall_survival'] * 100
    cleared = g['all_regions_cleared']
    print(f"  {label:<35} {ov_surv:>7.2f}% {bs_surv:>8.2f}% {cx_surv:>6.2f}% "
          f"{'YES' if cleared else 'NO':>8}")
    return ov_surv, bs_surv, cx_surv

# Baseline QD
run_spatial(73.6, 3.17, 48, 24, 'Baseline (QD, 48h delay)')
# BID
run_spatial(44.0, 10.82, 48, 12, 'BID (48h delay)')
# 72h delay + BID 
run_spatial(44.0, 10.82, 72, 12, '72h delay + BID')
# 96h delay + BID
run_spatial(44.0, 10.82, 96, 12, '96h delay + BID')
# 120h delay + BID (5 days!)
run_spatial(44.0, 10.82, 120, 12, '120h delay + BID')
# BID + half drug (EC50=12)
run_spatial(44.0, 10.82, 48, 12, 'BID + weak immune (rho=0.008)')
# BID + 2x daily dose
run_spatial(88.0, 21.64, 48, 12, 'BID + 2x dose')
# TID- best case
run_spatial(36.0, 15.56, 48, 8, 'TID (best case)')

print("\n" + "=" * 70)
print("  EVALUATION COMPLETE")
print("=" * 70)
