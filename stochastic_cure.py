"""
Hybrid ODE + Gillespie stochastic simulation.

Phase 1: deterministic ODE until the virus is suppressed (< 100 virions,
         < 10 infected cells).
Phase 2: Gillespie SSA from that suppressed state to check whether the residual
         infection goes extinct by chance or rebounds.

IMPORTANT: the hybrid phase-2 analysis is only meaningful if the deterministic
model actually reaches a suppressed state. If it never does, a stochastic
extinction calculation is not applicable and this script says so instead of
reporting a cure. Every parameter below is an assumed value, not a fit to rabies
data. See LIMITATIONS.md.
"""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
from pkpd_model import (
    simulate_pkpd_multi_dose, simulate_viral_dynamics_immune,
    VIRAL_BETA as BETA, VIRAL_DELTA as DELTA, VIRAL_P as P, VIRAL_C as CLR,
    VIRAL_EC50_UM as EC50_V, VIRAL_ALPHA as ALPHA, N_NEURONS,
)
from rdkit import Chem
from rdkit.Chem import Descriptors

# Single source of truth for the lead molecule.
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(_HERE, 'cure_found.json')) as f:
        smiles = json.load(f)['smiles']
except (OSError, KeyError, json.JSONDecodeError):
    smiles = 'CC1=CC(=O)NC2=CC=CC=C12'
print(f"Lead SMILES: {smiles}")

mol = Chem.MolFromSmiles(smiles)
mw = Descriptors.MolWt(mol)
daily_dose = 5e-3 * mw * 1000

# Get drug profile
pkpd = simulate_pkpd_multi_dose(smiles, dose_mg=daily_dose/2, n_doses=28, interval_hrs=12)
drug_t = pkpd['t']
drug_c = pkpd['C_brain_uM']

params = {
    'beta': BETA, 'delta': DELTA, 'p': P, 'c': CLR,
    'N0': N_NEURONS, 'alpha': ALPHA, 'EC50_v': EC50_V,
    'kappa': 0.003, 'sigma': 0.0005, 'rho': 0.015,
    'eta': 100.0, 'mu': 0.008, 'alpha_imm': 0.001, 'treatment_delay': 48.0,
}

def drug_conc(t):
    td = params['treatment_delay']
    if t < td:
        return 0.0
    return float(np.interp(t - td, drug_t, drug_c, left=0, right=0))

def drug_eff(D):
    if D <= 0:
        return 0.0
    ec50, alpha = params['EC50_v'], params['alpha']
    return alpha * D**1.5 / (ec50**1.5 + D**1.5)

# Phase 1: ODE until virus suppressed
print("=" * 65)
print("  HYBRID ODE + GILLESPIE STOCHASTIC SIMULATION")
print("  Lead molecule BID vs rabies (assumed parameters)")
print("=" * 65)
print()
print("  Phase 1: ODE until virus suppression...")

viral_ode = simulate_viral_dynamics_immune(pkpd, duration_hrs=720, treatment_delay_hrs=48)
S_ode = viral_ode['S']
I_ode = viral_ode['I']
V_ode = viral_ode['V']
E_ode = viral_ode['E']

# Find point where V < 100 and I < 10 (after day 14 to ensure drug is active)
switch_idx = None
for i in range(len(V_ode)):
    t = viral_ode['t'][i]
    if t > 400 and V_ode[i] < 100 and I_ode[i] < 10 and E_ode[i] > 1:
        switch_idx = i
        break

if switch_idx is None:
    V_final = V_ode[-1]
    I_final = I_ode[-1]
    print("  The deterministic model NEVER suppresses the infection"
          f" (final V={V_final:.0f}, I={I_final:.0f}).")
    print("  A stochastic extinction calculation is therefore not applicable:")
    print("  there is no suppressed state from which a residual infection could")
    print("  go extinct. Under these assumed parameters the model predicts a")
    print("  PERSISTENT infection, not a cure.")
    print()
    print("=" * 65)
    print("  CONCLUSION: no clearance under the assumed parameters.")
    print("  This is a model prediction, NOT a clinical claim.")
    print("=" * 65)
    sys.exit(0)

t_switch = viral_ode['t'][switch_idx]
S_init = int(round(S_ode[switch_idx]))
I_init = int(round(I_ode[switch_idx]))
V_init = int(round(V_ode[switch_idx]))
E_init = int(round(E_ode[switch_idx]))

print(f"  Switching at t={t_switch:.0f}h ({t_switch/24:.1f} days)")
print(f"  State: S={S_init}, I={I_init}, V={V_init}, E={E_init}")
print()

# Phase 2: Gillespie from there
def gillespie_step(state, t, dt_max=2.0):
    S, I, V, E = state
    D = drug_conc(t)
    de = drug_eff(D)
    p = params

    a = []
    reactions = []

    A = lambda s_: None  # placeholder

    # 1. Infection
    if S > 0 and V > 0:
        a.append(p['beta'] * S * V / p['N0'])
        reactions.append(lambda s: (s[0]-1, s[1]+1, s[2], s[3]))

    # 2. Infected cell death
    if I > 0:
        a.append(p['delta'] * I)
        reactions.append(lambda s: (s[0], s[1]-1, s[2], s[3]))

    # 3. Virus production
    if I > 0:
        a.append(p['p'] * I)
        reactions.append(lambda s: (s[0], s[1], s[2]+1, s[3]))

    # 4. Natural clearance
    if V > 0:
        a.append(p['c'] * V)
        reactions.append(lambda s: (s[0], s[1], s[2]-1, s[3]))

    # 5. Drug kill
    if V > 0 and de > 0:
        a.append(de * V)
        reactions.append(lambda s: (s[0], s[1], s[2]-1, s[3]))

    # 6. CTL kill
    if I > 0 and E > 0:
        a.append(p['kappa'] * E * I)
        reactions.append(lambda s: (s[0], s[1]-1, s[2], s[3]))

    # 7. CTL clear virus
    if V > 0 and E > 0:
        a.append(p['sigma'] * E * V)
        reactions.append(lambda s: (s[0], s[1], s[2]-1, s[3]))

    # 8. Basal immune
    a.append(p['alpha_imm'])
    reactions.append(lambda s: (s[0], s[1], s[2], s[3]+1))

    # 9. Immune decay
    if E > 0:
        a.append(p['mu'] * E)
        reactions.append(lambda s: (s[0], s[1], s[2], s[3]-1))

    # 10. Immune proliferation
    if E > 0 and V > 0:
        prolif = p['rho'] * V / (V + p['eta'])
        if prolif > 0:
            a.append(prolif * E)
            reactions.append(lambda s: (s[0], s[1], s[2], s[3]+1))

    a_total = sum(a)
    if a_total <= 0:
        return state, t, 'extinct'

    dt = np.random.exponential(1.0 / a_total)
    if dt > dt_max:
        dt = dt_max
        return state, t + dt, 'continue'

    r = np.random.random() * a_total
    cum = 0
    for i, ai in enumerate(a):
        cum += ai
        if r <= cum:
            state = reactions[i](state)
            break

    return state, t + dt, 'continue'


def run_gillespie(S0, I0, V0, E0, t0, T_max=17520, record_interval=168, seed=42):
    """Run Gillespie. Returns (times, states, outcome, peak_V_post_switch)."""
    np.random.seed(seed)
    state = (S0, I0, V0, E0)
    t = t0
    
    times = [t0]
    states = [state]
    peak_V = V0
    
    while True:
        state, t, outcome = gillespie_step(state, t, dt_max=2.0)
        S, I, V, E = state
        
        if V > peak_V:
            peak_V = V
        
        # Check extinction
        if V <= 0 and I <= 0:
            outcome = 'extinct'
            times.append(t)
            states.append(state)
            break
        
        # Check rebound: after suppression, virus growing back significantly
        if V > 10000 or I > 100:
            outcome = 'rebound'
            times.append(t)
            states.append(state)
            break
        
        if t - t0 > T_max:
            outcome = 'persistent'
            times.append(t)
            states.append(state)
            break
        
        # Record
        if t - times[-1] >= record_interval:
            times.append(t)
            states.append(state)
    
    return np.array(times), np.array(states), outcome, peak_V


# Run multiple trajectories
N = 200
print(f"  Phase 2: Running {N} Gillespie trajectories...")
print()

extinctions = 0
rebounds = 0
persistent = 0
peak_viruses = []
survivals = []
outcomes_list = []

for i in range(N):
    times, states, outcome, peak_V = run_gillespie(
        S_init, I_init, V_init, E_init, t_switch,
        seed=i*13 + 42
    )
    S_end = states[-1][0]
    I_end = states[-1][1]
    V_end = states[-1][2]
    
    survivals.append(S_end / 1_000_000 * 100)
    peak_viruses.append(peak_V)
    outcomes_list.append(outcome)
    
    if outcome == 'extinct':
        extinctions += 1
    elif outcome == 'rebound':
        rebounds += 1
    else:
        persistent += 1
    
    # Print progress
    if i < 5 or (i+1) % 50 == 0 or i == N-1:
        label = ("✓ EXTINCT" if outcome == 'extinct' 
                 else "✗ REBOUND" if outcome == 'rebound' 
                 else "? PERSISTS")
        print(f"  Traj {i+1:>3}: S={S_end:>7,} I={I_end:>4} V={V_end:>4} "
              f"peak_V={peak_V:>6} → {label}")

print()
print(f"  {'='*55}")
print(f"  RESULTS ({N} trajectories)")
print(f"  {'='*55}")
print(f"  Extinctions:  {extinctions:>3}/{N} ({extinctions/N*100:.1f}%)")
print(f"  Rebound:      {rebounds:>3}/{N} ({rebounds/N*100:.1f}%)")
print(f"  Persistent:   {persistent:>3}/{N} ({persistent/N*100:.1f}%)")
print(f"  Avg survival: {np.mean(survivals):.2f}%")
print(f"  Min survival: {min(survivals):.2f}%")
print(f"  Max survival: {max(survivals):.2f}%")
print(f"  Avg peak V:   {np.mean(peak_viruses):.1f}")
print()

# Summary — model output under assumed parameters only.
p_extinct = extinctions / N
print("  MODEL OUTPUT under assumed parameters — NOT a clinical claim.")
print("  'Extinct' = simulated virions and infected cells both reached zero.")
if p_extinct > 0.5:
    print(f"  CONCLUSION: stochastic clearance in {p_extinct*100:.0f}% of trajectories")
elif p_extinct > 0.1:
    print(f"  CONCLUSION: stochastic clearance in {p_extinct*100:.0f}% of trajectories")
    print(f"  Remaining trajectories rebound but reach {np.mean(survivals):.2f}% neuron survival")
else:
    print(f"  CONCLUSION: infection persists/rebounds in {(1-p_extinct)*100:.0f}% of trajectories")
    print(f"  Mean neuron survival {np.mean(survivals):.2f}% — survival of the abstract model")
    print(f"  does not imply viral clearance; see LIMITATIONS.md")
