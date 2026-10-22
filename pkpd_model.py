"""
PK/PD Model for Brain-Invading Virus Drug Delivery

Pharmacokinetics (PK):
- Two-compartment model: plasma → brain
- Drug crosses BBB based on permeability (from bbb_predictor)
- Parameters: absorption rate, distribution, elimination

Pharmacodynamics (PD):
- Viral suppression = Emax * C_brain / (EC50 + C_brain)
- Hill coefficient for dose-response curve

Usage:
    python pkpd_model.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --dose 500 --name "Aspirin"
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

try:
    import torch
    import torch.nn as nn
    from rdkit.Chem import Descriptors
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Simple BBB predictor (matches trained checkpoint)
BBB_MEAN = None
BBB_STD = None

class SimpleBBBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Physiological parameters (human averages)
PARAMS = {
    'ka': 1.0,          # absorption rate (1/hr) - oral
    'ke_plasma': 0.15,  # elimination from plasma (1/hr) - half-life ~4.6hr
    'ke_brain': 0.05,   # elimination from brain (1/hr) - slower clearance
    'V_plasma': 50.0,   # plasma volume (L)
    'V_brain': 1.5,     # brain volume (L)
    'Q': 2.0,           # blood flow to brain (L/hr)
    'bioavail': 0.7,    # oral bioavailability (fraction)
    'MW': 300.0,        # molecular weight for mg→mmol conversion
}

# --- Pharmacodynamic (PD) constants -----------------------------------------
# NOTE (scientific honesty): these values are NOT fitted to any experimental
# rabies/antiviral dataset. They are modelling assumptions. Every result that
# depends on them should be reported as "under assumed parameters", and
# sensitivity_analysis.py is provided to show how strongly the outcome depends
# on them. Changing EC50 alone moves the reported "cure" rate by tens of percent.
PD_EC50_UM = 10.0        # assumed brain EC50 for the lead chemotype (µM)
PD_HILL = 1.5            # assumed Hill coefficient
VIRAL_EC50_UM = 6.0      # assumed brain EC50 for free-virus suppression (µM)
VIRAL_ALPHA = 0.25       # assumed max drug kill rate on free virus
VIRAL_BETA = 0.012       # assumed neuron infection rate
VIRAL_DELTA = 0.02       # assumed infected-neuron death rate
VIRAL_P = 1.5            # assumed virion production per infected neuron
VIRAL_C = 0.4            # assumed free-virus clearance
N_NEURONS = 1e6

# A run is only called "cleared" if BOTH free virus and infected neurons fall to
# extinction and stay there (no rebound in the final 20% of the simulation).
EXTINCTION_THRESHOLD = 1.0


def get_bbb_permeability(smiles):
    """Get BBB permeability score from predictor. Returns pBBB (0-1)."""
    global BBB_MEAN, BBB_STD
    if not HAS_TORCH or not HAS_RDKIT:
        return 0.5
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5
        feat = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
        ]
        ckpt = torch.load('checkpoints_multivirus/bbb_predictor.pt', map_location='cpu', weights_only=True)
        model = SimpleBBBNet()
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        BBB_MEAN = ckpt['mean']
        BBB_STD = ckpt['std']
        x = torch.tensor([feat], dtype=torch.float32)
        x = (x - BBB_MEAN) / BBB_STD
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        return prob
    except Exception as e:
        print(f"  BBB prediction failed: {e}, using default 0.5")
        return 0.5


def pk_two_compartment(t, y, params):
    """Two-compartment PK ODE system.
    
    y[0] = amount in gut (oral dose absorption)
    y[1] = amount in plasma (mg)
    y[2] = amount in brain (mg)
    """
    Agut, Aplasma, Abrain = y
    ka = params['ka']
    ke_p = params['ke_plasma']
    ke_b = params['ke_brain']
    Vp = params['V_plasma']
    Vb = params['V_brain']
    Q = params['Q']
    P = params['pBBB']  # BBB permeability (0-1)

    # Absorption from gut
    dAgut = -ka * Agut

    # Plasma: absorption in - elimination - transfer to brain
    dAplasma = ka * Agut - ke_p * Aplasma - Q * P * (Aplasma / Vp - Abrain / Vb)

    # Brain: uptake from plasma - elimination
    dAbrain = Q * P * (Aplasma / Vp - Abrain / Vb) - ke_b * Abrain

    return [dAgut, dAplasma, dAbrain]


def pk_iv(t, y, params):
    """IV multi-dose PK: no gut, direct plasma injection.
    
    y[0] = amount in plasma (mg)
    y[1] = amount in brain (mg)
    """
    Aplasma, Abrain = y
    ke_p = params['ke_plasma']
    ke_b = params['ke_brain']
    Vp = params['V_plasma']
    Vb = params['V_brain']
    Q = params['Q']
    P = params['pBBB']

    dAplasma = -ke_p * Aplasma - Q * P * (Aplasma / Vp - Abrain / Vb)
    dAbrain = Q * P * (Aplasma / Vp - Abrain / Vb) - ke_b * Abrain

    return [dAplasma, dAbrain]


def simulate_pkpd(smiles, dose_mg=500, duration_hrs=48, dt=0.5, name="drug", params_override=None):
    """Run full PK/PD simulation for a drug candidate.
    
    Returns dict with time arrays, concentrations, and efficacy metrics.
    """
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
        else:
            mw = PARAMS['MW']
    else:
        mw = PARAMS['MW']

    pBBB = get_bbb_permeability(smiles)

    params = PARAMS.copy()
    params['pBBB'] = pBBB
    params['MW'] = mw
    if params_override:
        params.update(params_override)

    # Initial conditions: oral dose in gut, zero elsewhere
    dose_mmol = dose_mg / mw
    y0 = [dose_mmol * params['bioavail'], 0.0, 0.0]
    t_span = (0, duration_hrs)
    t_eval = np.arange(0, duration_hrs + dt, dt)

    sol = solve_ivp(
        pk_two_compartment, t_span, y0, args=(params,),
        t_eval=t_eval, method='RK45', dense_output=True,
        rtol=1e-6, atol=1e-9
    )

    Agut, Aplasma, Abrain = sol.y

    # Concentrations in mmol/L
    C_plasma = Aplasma / params['V_plasma']
    C_brain = Abrain / params['V_brain']

    # Convert to μM for easier reading (1 mmol/L = 1000 μM)
    C_plasma_uM = C_plasma * 1000
    C_brain_uM = C_brain * 1000

    # PD: viral suppression (assumed EC50 / Hill; see constants at top of file)
    EC50 = PD_EC50_UM
    Emax = 1.0
    hill = PD_HILL
    suppression = Emax * C_brain_uM**hill / (EC50**hill + C_brain_uM**hill)

    # Key PK metrics
    Cmax_plasma = C_plasma_uM.max()
    Cmax_brain = C_brain_uM.max()
    Tmax_plasma = t_eval[C_plasma_uM.argmax()]
    Tmax_brain = t_eval[C_brain_uM.argmax()]

    # AUC (trapezoidal)
    AUC_plasma = np.trapz(C_plasma_uM, t_eval)
    AUC_brain = np.trapz(C_brain_uM, t_eval)

    # Time above EC50
    time_above_ec50 = np.sum(C_brain_uM >= EC50) * dt

    # Maximum suppression
    max_suppression = suppression.max()

    # Average suppression over 24hr
    idx_24hr = t_eval <= 24
    avg_suppression_24h = suppression[idx_24hr].mean() if idx_24hr.any() else 0

    result = {
        'name': name,
        'smiles': smiles,
        'dose_mg': dose_mg,
        'pBBB': pBBB,
        'mw': mw,
        't': t_eval,
        'C_plasma_uM': C_plasma_uM,
        'C_brain_uM': C_brain_uM,
        'suppression': suppression,
        'Cmax_plasma_uM': Cmax_plasma,
        'Cmax_brain_uM': Cmax_brain,
        'Tmax_plasma_hrs': Tmax_plasma,
        'Tmax_brain_hrs': Tmax_brain,
        'AUC_plasma': AUC_plasma,
        'AUC_brain': AUC_brain,
        'time_above_ec50_hrs': time_above_ec50,
        'max_suppression': max_suppression,
        'avg_suppression_24h': avg_suppression_24h,
        'half_life_plasma_hrs': np.log(2) / params['ke_plasma'],
        'half_life_brain_hrs': np.log(2) / params['ke_brain'],
    }

    return result


def simulate_pkpd_multi_dose(smiles, dose_mg=500, interval_hrs=24, n_doses=14, name="drug", params_override=None):
    """IV multi-dose PK/PD: daily dosing for n_doses days.
    
    Returns same format as simulate_pkpd but with sustained brain concentration.
    """
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
        else:
            mw = PARAMS['MW']
    else:
        mw = PARAMS['MW']

    pBBB = get_bbb_permeability(smiles)

    params = PARAMS.copy()
    params['pBBB'] = pBBB
    params['MW'] = mw
    if params_override:
        params.update(params_override)

    total_hrs = interval_hrs * n_doses + 48  # extra 48h to see washout
    dt = 0.5
    t_eval = np.arange(0, total_hrs + dt, dt)

    # Solve piecewise: add dose at start of each interval
    A_plasma = 0.0
    A_brain = 0.0
    all_Cp = []
    all_Cb = []

    dose_mmol = dose_mg / mw

    for dose_idx in range(n_doses):
        t_start = dose_idx * interval_hrs
        t_end = min((dose_idx + 1) * interval_hrs, total_hrs)
        t_span = (t_start, t_end)
        t_seg = np.arange(t_start, t_end + dt, dt)

        # Add IV dose directly to plasma
        A_plasma += dose_mmol

        sol = solve_ivp(
            pk_iv, t_span, [A_plasma, A_brain],
            args=(params,),
            t_eval=t_seg, method='RK45', rtol=1e-4, atol=1e-6
        )

        A_plasma = sol.y[0, -1]
        A_brain = sol.y[1, -1]

        Cp_seg = sol.y[0] / params['V_plasma'] * 1000  # μM
        Cb_seg = sol.y[1] / params['V_brain'] * 1000

        if dose_idx == 0:
            all_Cp = list(Cp_seg)
            all_Cb = list(Cb_seg)
        else:
            all_Cp.extend(list(Cp_seg[1:]))
            all_Cb.extend(list(Cb_seg[1:]))

    all_Cp = np.array(all_Cp)
    all_Cb = np.array(all_Cb)
    t_out = np.linspace(0, len(all_Cb) * dt, len(all_Cb))

    # PD (assumed parameters)
    EC50 = PD_EC50_UM
    Emax = 1.0
    hill = PD_HILL
    suppression = Emax * all_Cb**hill / (EC50**hill + all_Cb**hill)

    # Metrics
    Cmax_brain = all_Cb.max()
    AUC_brain = np.trapz(all_Cb, t_out)
    time_above_ec50 = np.sum(all_Cb >= EC50) * dt
    avg_suppression = suppression.mean()

    # Steady-state trough concentration (last dose)
    trough_brain = all_Cb[-int(24/dt):].min() if len(all_Cb) > int(24/dt) else all_Cb.min()

    result = {
        'name': name,
        'smiles': smiles,
        'dose_mg': dose_mg,
        'pBBB': pBBB,
        'mw': mw,
        'n_doses': n_doses,
        'interval_hrs': interval_hrs,
        't': t_out,
        'C_plasma_uM': all_Cp,
        'C_brain_uM': all_Cb,
        'suppression': suppression,
        'Cmax_brain_uM': Cmax_brain,
        'AUC_brain': AUC_brain,
        'time_above_ec50_hrs': time_above_ec50,
        'max_suppression': suppression.max(),
        'avg_suppression': avg_suppression,
        'trough_brain_uM': trough_brain,
        'half_life_plasma_hrs': np.log(2) / params['ke_plasma'],
        'half_life_brain_hrs': np.log(2) / params['ke_brain'],
    }

    return result


def simulate_pkpd_infusion(smiles, total_daily_mg=500, infusion_hrs=24, n_days=14, name="drug", params_override=None):
    """Continuous IV infusion PK/PD: steady drip instead of bolus doses.

    Drug concentration reaches steady state and stays constant,
    maintaining brain concentration above EC50 throughout treatment.
    """
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol) if mol else PARAMS['MW']
    else:
        mw = PARAMS['MW']

    pBBB = get_bbb_permeability(smiles)

    params = PARAMS.copy()
    params['pBBB'] = pBBB
    params['MW'] = mw
    if params_override:
        params.update(params_override)

    total_hrs = n_days * 24 + 48
    dt = 0.25
    t_eval = np.arange(0, total_hrs + dt, dt)

    # Infusion rate (mmol/hr)
    dose_mmol = total_daily_mg / mw
    infusion_rate = dose_mmol / infusion_hrs  # mmol/hr

    # PK with constant infusion
    A_plasma, A_brain = 0.0, 0.0
    all_Cp, all_Cb = [], []

    ke_p = params['ke_plasma']
    ke_b = params['ke_brain']
    Vp = params['V_plasma']
    Vb = params['V_brain']
    Q = params['Q']
    P = params['pBBB']

    for t in t_eval:
        day_frac = t / 24
        is_infusing = day_frac < n_days and (t % 24) < infusion_hrs

        if is_infusing:
            dAplasma = infusion_rate - ke_p * A_plasma - Q * P * (A_plasma / Vp - A_brain / Vb)
        else:
            dAplasma = -ke_p * A_plasma - Q * P * (A_plasma / Vp - A_brain / Vb)

        dAbrain = Q * P * (A_plasma / Vp - A_brain / Vb) - ke_b * A_brain

        A_plasma += dAplasma * dt
        A_brain += dAbrain * dt

        # Clamp
        A_plasma = max(0, A_plasma)
        A_brain = max(0, A_brain)

        Cp = A_plasma / Vp * 1000
        Cb = A_brain / Vb * 1000
        all_Cp.append(Cp)
        all_Cb.append(Cb)

    all_Cp = np.array(all_Cp)
    all_Cb = np.array(all_Cb)

    # PD (assumed parameters)
    EC50 = PD_EC50_UM
    Emax = 1.0
    hill = PD_HILL
    suppression = Emax * all_Cb**hill / (EC50**hill + all_Cb**hill)

    Cmax_brain = all_Cb.max()
    AUC_brain = np.trapz(all_Cb, t_eval)
    time_above_ec50 = np.sum(all_Cb >= EC50) * dt
    avg_suppression = suppression.mean()

    # Trough during steady-state (last day of infusion)
    infusion_end_idx = int(n_days * 24 / dt)
    steady_start = infusion_end_idx - int(24 / dt)
    if steady_start >= 0:
        trough_brain = all_Cb[steady_start:infusion_end_idx].min()
    else:
        trough_brain = all_Cb.min()

    result = {
        'name': name,
        'smiles': smiles,
        'dose_mg': total_daily_mg,
        'pBBB': pBBB,
        'mw': mw,
        'infusion_hrs': infusion_hrs,
        'n_days': n_days,
        'n_doses': n_days,
        'interval_hrs': 24,
        't': t_eval,
        'C_plasma_uM': all_Cp,
        'C_brain_uM': all_Cb,
        'suppression': suppression,
        'Cmax_brain_uM': Cmax_brain,
        'AUC_brain': AUC_brain,
        'time_above_ec50_hrs': time_above_ec50,
        'max_suppression': suppression.max(),
        'avg_suppression': avg_suppression,
        'trough_brain_uM': trough_brain.min(),
        'half_life_plasma_hrs': np.log(2) / ke_p,
        'half_life_brain_hrs': np.log(2) / ke_b,
    }

    return result


def print_pkpd_report(result):
    """Print formatted PK/PD report."""
    r = result
    print(f"\n{'='*60}")
    print(f"  PK/PD REPORT: {r['name']}")
    print(f"{'='*60}")
    print(f"  SMILES:       {r['smiles']}")
    print(f"  Dose:         {r['dose_mg']} mg")
    print(f"  MW:           {r['mw']:.1f} g/mol")
    print(f"  BBB perm:     {r['pBBB']:.3f} ({'High' if r['pBBB']>0.7 else 'Moderate' if r['pBBB']>0.3 else 'Low'})")
    print(f"  Half-life:    {r['half_life_plasma_hrs']:.1f}h (plasma), {r['half_life_brain_hrs']:.1f}h (brain)")
    print(f"\n  --- Plasma PK ---")
    print(f"  Cmax:         {r['Cmax_plasma_uM']:.1f} μM at {r['Tmax_plasma_hrs']:.1f}h")
    print(f"  AUC:          {r['AUC_plasma']:.0f} μM·hr")
    print(f"\n  --- Brain PK ---")
    print(f"  Cmax:         {r['Cmax_brain_uM']:.1f} μM at {r['Tmax_brain_hrs']:.1f}h")
    print(f"  AUC:          {r['AUC_brain']:.0f} μM·hr")
    print(f"\n  --- PD (Viral Suppression) ---")
    print(f"  Max suppression:     {r['max_suppression']:.1%}")
    print(f"  Avg suppression 24h: {r['avg_suppression_24h']:.1%}")
    print(f"  Time above EC50:     {r['time_above_ec50_hrs']:.1f} hrs")
    print(f"{'='*60}\n")


# --- Viral dynamics model ---

def viral_dynamics(t, y, params):
    """SIR-like model for rabies in neural tissue.
    
    y[0] = S (susceptible neurons)
    y[1] = I (infected neurons)
    y[2] = V (free virus particles)
    y[3] = D (drug concentration in brain, μM)
    """
    S, I, V, D = y

    beta = params.get('beta', 0.5)       # infection rate
    delta = params.get('delta', 0.1)     # infected cell death rate
    p = params.get('p', 10.0)            # virus production rate per infected cell
    c = params.get('c', 1.0)             # virus clearance rate
    N0 = params.get('N0', 1e6)           # initial neuron count
    alpha = params.get('alpha', 0.02)    # drug efficacy (max viral kill rate)
    EC50_v = params.get('EC50_v', 5.0)   # EC50 for viral suppression (μM)

    # Drug efficacy (Hill equation)
    drug_eff = alpha * D**1.5 / (EC50_v**1.5 + D**1.5)

    dS = -beta * S * V / N0
    dI = beta * S * V / N0 - delta * I
    dV = p * I - c * V - drug_eff * V  # drug kills free virus
    dD = -0.05 * D  # brain drug elimination

    return [dS, dI, dV, dD]


def viral_dynamics_immune(t, y, params):
    """Viral dynamics WITH adaptive immune response.
    
    y[0] = S (susceptible neurons)
    y[1] = I (infected neurons)
    y[2] = V (free virus particles)
    y[3] = E (immune effectors, e.g. CTLs)
    y[4] = D (drug concentration in brain, μM)
    """
    S, I, V, E, D = y

    beta = params.get('beta', 0.012)
    delta = params.get('delta', 0.02)
    p = params.get('p', 1.5)
    c = params.get('c', 0.4)
    N0 = params.get('N0', 1e6)
    alpha = params.get('alpha', 0.25)
    EC50_v = params.get('EC50_v', 6.0)

    # Immune parameters
    kappa = params.get('kappa', 0.005)     # CTL killing of infected cells
    sigma = params.get('sigma', 0.001)     # CTL viral clearance
    rho = params.get('rho', 0.02)          # max CTL proliferation rate
    eta = params.get('eta', 100.0)         # half-saturation for CTL activation
    mu = params.get('mu', 0.005)           # CTL decay rate
    alpha_imm = params.get('alpha_imm', 0.001)  # basal CTL production

    drug_eff = alpha * D**1.5 / (EC50_v**1.5 + D**1.5)

    dS = -beta * S * V / N0
    dI = beta * S * V / N0 - delta * I - kappa * E * I
    dV = p * I - c * V - drug_eff * V - sigma * E * V
    dE = alpha_imm + rho * E * V / (V + eta) - mu * E
    dD = -0.05 * D

    return [dS, dI, dV, dE, dD]


def classify_outcome(t, S, I, V, n0=N_NEURONS, rebound_window=0.2):
    """Classify a viral-dynamics run honestly.

    Previously the code called a run "cured" whenever >99% of neurons survived,
    even if virus and infected cells were still present at the end. That is
    tissue preservation, not sterilization. This helper separates the two:

      cleared     — free virus AND infected cells driven to extinction, with no
                    rebound (they stay below threshold for the final 20% of the
                    simulation). This is the only defensible meaning of "cure".
      suppressed  — viral load stays below its peak throughout the tail, i.e. the
                    drug holds the infection down without eliminating it.
      survival_fraction — fraction of the modelled neurons still alive at the end.

    Returns a dict merged into the simulation result.
    """
    n = len(t)
    tail = slice(int((1 - rebound_window) * n), None)
    final_virus = float(V[-1])
    final_infected = float(I[-1])

    extinct_now = final_virus < EXTINCTION_THRESHOLD and final_infected < EXTINCTION_THRESHOLD
    no_rebound = bool(V[tail].max() < EXTINCTION_THRESHOLD and
                      I[tail].max() < EXTINCTION_THRESHOLD)
    cleared = bool(extinct_now and no_rebound)

    peak = float(V.max())
    suppressed = bool(V[tail].max() <= max(0.05 * peak, EXTINCTION_THRESHOLD))

    return {
        'cleared': cleared,
        'suppressed': suppressed,
        'final_virus': final_virus,
        'final_infected': final_infected,
        'survival_fraction': float(S[-1] / n0),
    }


def _clearance_label(viral):
    """Map a viral-dynamics result to a human-readable clearance status."""
    if viral.get('cleared'):
        return 'CLEARED (virus + infected cells extinct, no rebound)'
    if viral.get('suppressed'):
        return 'SUPPRESSED (held below peak, not eliminated)'
    return 'ONGOING (fails to control infection)'


def simulate_viral_dynamics(drug_result, initial_virus=1000, duration_hrs=None, treatment_delay_hrs=48):
    """Simulate rabies spread with drug treatment.
    
    treatment_delay_hrs: time between infection and drug administration
    (simulates real clinical scenario - patient shows symptoms late)
    
    If duration_hrs is None, auto-detect from drug_result length.
    """
    if duration_hrs is None:
        drug_duration = drug_result['t'][-1]
        duration_hrs = max(168, drug_duration + treatment_delay_hrs + 72)  # extra 3 days after last dose
    
    t_eval = np.arange(0, int(duration_hrs) + 1, 1)

    # Interpolate brain drug concentration
    drug_t = drug_result['t']
    drug_c = drug_result['C_brain_uM']

    v_params = {
        'beta': VIRAL_BETA,
        'delta': VIRAL_DELTA,
        'p': VIRAL_P,
        'c': VIRAL_C,
        'N0': N_NEURONS,
        'alpha': VIRAL_ALPHA,
        'EC50_v': VIRAL_EC50_UM,
    }

    def ode_with_drug(t, y):
        S, I, V, D_drug = y
        # Drug only present after treatment delay
        if t < treatment_delay_hrs:
            D_val = 0.0
        else:
            D_val = np.interp(t - treatment_delay_hrs, drug_t, drug_c)
        return viral_dynamics(t, [S, I, V, D_val], v_params)

    y0 = [1e6, 1, initial_virus, 0]
    sol = solve_ivp(ode_with_drug, (0, float(duration_hrs)), y0, t_eval=t_eval, method='RK45', rtol=1e-3, atol=1e-6)

    S, I, V, D = sol.y

    outcome = classify_outcome(t_eval, S, I, V)
    result = {
        't': t_eval,
        'S': S, 'I': I, 'V': V,
        'peak_virus': float(V.max()),
        'peak_virus_time_hrs': float(t_eval[V.argmax()]),
        'neurons_saved': float(S[-1]),
        # 'cured' kept for backward compatibility but now means viral CLEARANCE
        # (see classify_outcome), not ">99% of neurons survived".
        'cured': outcome['cleared'],
        **outcome,
    }

    return result


def simulate_viral_dynamics_immune(drug_result, initial_virus=1000, duration_hrs=None,
                                    treatment_delay_hrs=48, immune_params=None):
    """Simulate rabies spread WITH adaptive immune response + drug.

    The immune system helps clear infected cells and free virus,
    making cure possible even with imperfect drug suppression.
    """
    if duration_hrs is None:
        drug_duration = drug_result['t'][-1]
        duration_hrs = max(168, drug_duration + treatment_delay_hrs + 168)

    t_eval = np.arange(0, int(duration_hrs) + 1, 1)
    drug_t = drug_result['t']
    drug_c = drug_result['C_brain_uM']

    v_params = {
        'beta': VIRAL_BETA, 'delta': VIRAL_DELTA, 'p': VIRAL_P, 'c': VIRAL_C,
        'N0': N_NEURONS, 'alpha': VIRAL_ALPHA, 'EC50_v': VIRAL_EC50_UM,
        'kappa': 0.003, 'sigma': 0.0005, 'rho': 0.015,
        'eta': 100.0, 'mu': 0.008, 'alpha_imm': 0.001,
    }
    if immune_params:
        v_params.update(immune_params)

    def ode_fn(t, y):
        S, I, V, E, D_drug = y
        if t < treatment_delay_hrs:
            D_val = 0.0
        else:
            D_val = np.interp(t - treatment_delay_hrs, drug_t, drug_c)
        return viral_dynamics_immune(t, [S, I, V, E, D_val], v_params)

    E0 = v_params['alpha_imm'] / v_params['mu']
    y0 = [1e6, 1, initial_virus, E0, 0]

    sol = solve_ivp(ode_fn, (0, float(duration_hrs)), y0, t_eval=t_eval,
                    method='RK45', rtol=1e-3, atol=1e-6)

    S, I, V, E, D = sol.y
    outcome = classify_outcome(t_eval, S, I, V)
    result = {
        't': t_eval,
        'S': S, 'I': I, 'V': V, 'E': E,
        'peak_virus': float(V.max()),
        'peak_virus_time_hrs': float(t_eval[V.argmax()]),
        'neurons_saved': float(S[-1]),
        'immune_peak': float(E.max()),
        # 'cured' now means viral CLEARANCE (see classify_outcome).
        'cured': outcome['cleared'],
        **outcome,
    }

    return result


# --- Toxicity predictor ---

TOX_FEATURES = [
    'hERG_block',       # cardiac toxicity
    'hepatotox',        # liver toxicity
    'amx_mutagen',      # mutagenicity
    'LD50_log',         # acute toxicity
    'CYP_inhib',        # drug interaction
]


def predict_toxicity(smiles):
    """Rule-based toxicity flags for drug candidates."""
    if not HAS_RDKIT:
        return {'toxic': False, 'flags': [], 'score': 0.0}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'toxic': True, 'flags': ['invalid_smiles'], 'score': 1.0}

    flags = []
    score = 0.0

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    alerts = Descriptors.NumAromaticRings(mol)

    # Lipinski violations (overdose risk)
    if mw > 500:
        flags.append('high_mw')
        score += 0.15
    if logp > 5:
        flags.append('high_logp')
        score += 0.15
    if logp < -1:
        flags.append('very_hydrophilic')
        score += 0.1

    # hERG risk (cardiac)
    if logp > 3 and hba > 2 and mw < 500:
        flags.append('herg_risk')
        score += 0.25

    # Reactive groups
    smi_upper = smiles.upper()
    if 'N=N' in smi_upper or 'N=N' in smiles:
        flags.append('azo_group')
        score += 0.3
    if 'C(=O)Cl' in smiles or 'C(=O)Br' in smiles:
        flags.append('acyl_halide')
        score += 0.5
    if 'C1=CC=C(N)C=C1' in smiles or 'C1=CC=C(N)C=C1':
        flags.append('aniline')
        score += 0.1

    # Peptide instability
    if 'NC(=O)' in smiles and smiles.count('NC(=O)') > 3:
        flags.append('polyptide')
        score += 0.15

    # CYP inhibition (aromatic + lipophilic)
    if alerts >= 2 and logp > 2:
        flags.append('cyp_inhibition_risk')
        score += 0.15

    # Overall toxicity score
    score = min(score, 1.0)

    return {
        'toxic': score > 0.3,
        'flags': flags,
        'score': score,
        'herg_risk': any('herg' in f for f in flags),
        'hepatotox_risk': score > 0.4,
        'safe': score < 0.2,
    }


def score_candidate(smiles, name="candidate", dose_mg=500):
    """Full PK/PD/toxicity scoring for a drug candidate."""
    print(f"\n{'#'*60}")
    print(f"  SCORING: {name}")
    print(f"{'#'*60}")

    # 1. BBB
    pBBB = get_bbb_permeability(smiles)
    print(f"  BBB permeability: {pBBB:.3f}")

    # 2. PK/PD
    pkpd = simulate_pkpd(smiles, dose_mg=dose_mg, name=name)
    print_pkpd_report(pkpd)

    # 3. Viral dynamics (48hr treatment delay - realistic for rabies)
    viral = simulate_viral_dynamics(pkpd, treatment_delay_hrs=48)
    status = _clearance_label(viral)
    print(f"  Viral dynamics: peak={viral['peak_virus']:.0f} at {viral['peak_virus_time_hrs']:.0f}h")
    print(f"  Neurons preserved: {viral['survival_fraction']:.1%}")
    print(f"  Clearance: {status} "
          f"(final virus={viral['final_virus']:.2g}, infected cells={viral['final_infected']:.2g})")

    # 4. Toxicity
    tox = predict_toxicity(smiles)
    print(f"  Toxicity: {'TOXIC' if tox['toxic'] else 'SAFE'} (score={tox['score']:.2f})")
    if tox['flags']:
        print(f"  Flags: {', '.join(tox['flags'])}")

    # 5. Heuristic ranking score (NOT a scientific result). The weights below are an
    # arbitrary prioritisation used only to order candidates; each component is also
    # reported separately above so the reader can judge them on their own terms.
    bbb_score = pBBB
    pk_score = pkpd['avg_suppression_24h']
    neuro_score = viral['survival_fraction']
    tox_score = 1.0 - tox['score']

    composite = (bbb_score * 0.25 + pk_score * 0.25 + neuro_score * 0.30 + tox_score * 0.20)
    print(f"\n  Heuristic ranking score: {composite:.3f}  (arbitrary weighting, for ordering only)")
    print(f"    BBB:             {bbb_score:.3f} (25%)")
    print(f"    PK/PD suppr.:    {pk_score:.3f} (25%)")
    print(f"    Neuroprotection: {neuro_score:.3f} (30%)")
    print(f"    Safety:          {tox_score:.3f} (20%)")

    return {
        'name': name,
        'smiles': smiles,
        'pBBB': pBBB,
        'pkpd': pkpd,
        'viral': viral,
        'toxicity': tox,
        'composite_score': composite,
    }


def score_candidate_multi_dose(smiles, name="candidate", dose_mg=500, n_doses=14, interval_hrs=24, verbose=True):
    """Full PK/PD/toxicity scoring WITH multi-dose IV treatment."""
    if verbose:
        print(f"\n{'#'*60}")
        print(f"  SCORING (multi-dose): {name}")
        print(f"{'#'*60}")

    pBBB = get_bbb_permeability(smiles)
    if verbose:
        print(f"  BBB permeability: {pBBB:.3f}")

    pkpd = simulate_pkpd_multi_dose(smiles, dose_mg=dose_mg, n_doses=n_doses, interval_hrs=interval_hrs, name=name)
    if verbose:
        print(f"  Daily dose: {dose_mg}mg x {n_doses} days (IV)")
        print(f"  Brain Cmax:  {pkpd['Cmax_brain_uM']:.1f} uM")
        print(f"  Trough:      {pkpd['trough_brain_uM']:.1f} uM")
        print(f"  Avg suppr:   {pkpd['avg_suppression']:.1%}")

    viral = simulate_viral_dynamics(pkpd, treatment_delay_hrs=48)
    if verbose:
        print(f"  Neurons preserved: {viral['survival_fraction']:.1%}")
        print(f"  Clearance:    {_clearance_label(viral)}")

    tox = predict_toxicity(smiles)
    if verbose:
        print(f"  Toxicity:    {'TOXIC' if tox['toxic'] else 'SAFE'} (score={tox['score']:.2f})")

    bbb_score = pBBB
    pk_score = pkpd['avg_suppression']
    neuro_score = viral['survival_fraction']
    tox_score = 1.0 - tox['score']

    composite = bbb_score * 0.25 + pk_score * 0.25 + neuro_score * 0.30 + tox_score * 0.20

    return {
        'name': name,
        'smiles': smiles,
        'pBBB': pBBB,
        'pkpd': pkpd,
        'viral': viral,
        'toxicity': tox,
        'composite_score': composite,
    }


if __name__ == '__main__':
    # Test known drugs
    test_drugs = [
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin', 500),
        ('CN(C)CCCC(c1ccc(F)cc1)c1ccccc1', 'Fluoxetine (Prozac)', 20),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine', 200),
        ('CC(=O)NC1=CC=CC=C1', 'Acetaminophen', 500),
        ('CC(C)Cc1ccc(C(C)C(=O)O)cc1', 'Ibuprofen', 400),
        ('CN(C)CC(CO)c1ccc(O)cc1', 'Salbutamol', 4),
        ('OC(=O)C1=CC=CC=C1O', 'Salicylic acid', 500),
    ]

    results = []
    for smi, name, dose in test_drugs:
        r = score_candidate(smi, name=name, dose_mg=dose)
        results.append(r)

    # Summary ranking
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    print(f"\n{'='*60}")
    print(f"  CANDIDATE RANKING")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:<25} Score: {r['composite_score']:.3f}  BBB: {r['pBBB']:.2f}  Toxic: {r['toxicity']['toxic']}")
    print(f"{'='*60}")
