"""
Spatial Multi-Compartment Brain Model for Rabies Spread

Rabies spreads along neural pathways, not uniformly.
This model tracks infection + drug across connected brain regions.

Connectivity: Entry → SpinalCord → Brainstem → Thalamus → Cortex
                                            ↙ ↘
                                      Cerebellum Hippocampus
"""
import numpy as np
from scipy.integrate import solve_ivp

# Brain regions with properties
REGIONS = {
    'PNS_Entry': {
        'name': 'Peripheral entry site',
        'neurons': 1e3,
        'bbb_perm': 0.01,
        'color': '#ff6b6b',
    },
    'SpinalCord': {
        'name': 'Spinal cord (first CNS entry)',
        'neurons': 5e5,
        'bbb_perm': 0.60,
        'color': '#ffa502',
    },
    'Brainstem': {
        'name': 'Brainstem (breathing/heart rate)',
        'neurons': 5e5,
        'bbb_perm': 0.55,
        'color': '#eccc68',
    },
    'Thalamus': {
        'name': 'Thalamus (sensory relay)',
        'neurons': 1e6,
        'bbb_perm': 0.45,
        'color': '#2ed573',
    },
    'Cerebellum': {
        'name': 'Cerebellum (coordination)',
        'neurons': 5e6,
        'bbb_perm': 0.40,
        'color': '#70a1ff',
    },
    'Hippocampus': {
        'name': 'Hippocampus (memory/emotion)',
        'neurons': 1e6,
        'bbb_perm': 0.50,
        'color': '#a29bfe',
    },
    'Hypothalamus': {
        'name': 'Hypothalamus (autonomic)',
        'neurons': 1e5,
        'bbb_perm': 0.45,
        'color': '#fd79a8',
    },
    'BasalGanglia': {
        'name': 'Basal ganglia (movement)',
        'neurons': 1e6,
        'bbb_perm': 0.45,
        'color': '#e17055',
    },
    'Cortex': {
        'name': 'Cerebral cortex (higher function)',
        'neurons': 5e6,
        'bbb_perm': 0.30,
        'color': '#636e72',
    },
}

# Neural connectivity: source → [targets] with transport delays (hours)
CONNECTIVITY = {
    'PNS_Entry':  [('SpinalCord', 6)],
    'SpinalCord': [('Brainstem', 12), ('Cerebellum', 18)],
    'Brainstem':  [('Thalamus', 6), ('Hypothalamus', 4), ('Cerebellum', 8)],
    'Thalamus':   [('Hippocampus', 4), ('BasalGanglia', 6), ('Cortex', 8)],
    'Hippocampus': [('Cortex', 4)],
    'BasalGanglia': [('Cortex', 6)],
    'Hypothalamus': [('Thalamus', 4), ('Brainstem', 4)],
    'Cerebellum': [],
    'Cortex': [],
}

VIRAL_TRANSPORT_RATE = 0.005

# A region counts as "cleared" only if free virus AND infected cells are driven to
# extinction and stay below this threshold in the final 20% of the simulation.
# (>99% neuronal survival alone is tissue preservation, not sterilization.)
EXTINCTION_THRESHOLD = 1.0
REBOUND_WINDOW = 0.2

# Order for solving (upstream → downstream)
REGION_ORDER = ['PNS_Entry', 'SpinalCord', 'Brainstem', 'Hypothalamus',
                'Thalamus', 'Cerebellum', 'Hippocampus', 'BasalGanglia', 'Cortex']


def viral_dynamics_region(t, y, params, drug_func=None):
    """S-I-V-E-D ODE for a single brain region.
    
    y = [S, I, V, E, D] for this region
    """
    S, I, V, E, D = y
    beta = params.get('beta', 0.012)
    delta = params.get('delta', 0.02)
    p = params.get('p', 1.5)
    c = params.get('c', 0.4)
    N0 = params.get('N0', 1e6)
    alpha = params.get('alpha', 0.25)
    EC50_v = params.get('EC50_v', 6.0)
    kappa = params.get('kappa', 0.003)
    sigma = params.get('sigma', 0.0005)
    rho = params.get('rho', 0.015)
    eta = params.get('eta', 100.0)
    mu = params.get('mu', 0.008)
    alpha_imm = params.get('alpha_imm', 0.001)

    drug_eff = alpha * D**1.5 / (EC50_v**1.5 + D**1.5) if D > 0 else 0

    dS = -beta * S * V / N0
    dI = beta * S * V / N0 - delta * I - kappa * E * I
    dV = p * I - c * V - drug_eff * V - sigma * E * V
    dE = alpha_imm + rho * E * V / (V + eta) - mu * E
    dD = -0.05 * D + (drug_func(t) if drug_func else 0)

    return [dS, dI, dV, dE, dD]


def simulate_spatial_rabies(drug_cmax=73.6, drug_trough=3.17,
                            treatment_delay_hrs=48, dose_interval_hrs=24,
                            total_hrs=720, drug_half_life=6.0):
    """Simulate rabies spread through connected brain regions with drug treatment.
    
    drug_cmax: peak brain concentration (µM) after each dose
    drug_trough: minimum brain concentration (µM) before next dose
    treatment_delay_hrs: when first dose is given (hours post-infection)
    dose_interval_hrs: hours between doses (24=QD, 12=BID, 8=TID)
    """
    ke = np.log(2) / drug_half_life  # elimination rate

    def drug_concentration(t):
        """Drug concentration at time t in plasma.
        
        Models periodic IV bolus with exponential decay.
        """
        if t < treatment_delay_hrs:
            return 0.0
        elapsed = t - treatment_delay_hrs
        dose_idx = int(elapsed / dose_interval_hrs)
        time_since_dose = elapsed - dose_idx * dose_interval_hrs
        # Exponential decay: C = Cmax * exp(-ke * t)
        # Trough = Cmax * exp(-ke * interval), so Cmax = trough / exp(-ke * interval)
        cpeak = drug_trough / np.exp(-ke * dose_interval_hrs)
        conc = cpeak * np.exp(-ke * time_since_dose)
        return max(0, conc)

    v_params = {
        'beta': 0.012, 'delta': 0.02, 'p': 1.5, 'c': 0.4,
        'alpha': 0.25, 'EC50_v': 6.0,
        'kappa': 0.003, 'sigma': 0.0005, 'rho': 0.015,
        'eta': 100.0, 'mu': 0.008, 'alpha_imm': 0.001,
    }

    t_eval = np.arange(0, total_hrs + 1, 1)
    n_regions = len(REGION_ORDER)
    y0 = []

    for region in REGION_ORDER:
        info = REGIONS[region]
        N0 = info['neurons']
        if region == 'PNS_Entry':
            # Infection starts at peripheral entry
            y0.extend([N0 - 1, 1, 100, 0.125, 0.0])
        else:
            y0.extend([N0, 0, 0, 0.125, 0.0])

    def ode_system(t, y):
        dydt = []
        y_reshaped = y.reshape(n_regions, 5)

        # Full ODE per region
        for i, region in enumerate(REGION_ORDER):
            info = REGIONS[region]
            vp = v_params.copy()
            vp['N0'] = info['neurons']
            bbb = info['bbb_perm']
            S_i, I_i, V_i, E_i, D_i = y_reshaped[i]

            # Drug enters region via BBB (fraction of plasma concentration)
            plasma_drug = drug_concentration(t)
            drug_influx = bbb * 0.1 * (plasma_drug - D_i)

            dd = viral_dynamics_region(t, y_reshaped[i], vp)
            dd[4] = drug_influx - 0.05 * D_i

            # Add viral inflow from upstream regions (continuous neural spread)
            for src, targets in CONNECTIVITY.items():
                if region in [t[0] for t in targets]:
                    src_idx = REGION_ORDER.index(src)
                    V_src = y_reshaped[src_idx, 2]
                    delay_info = [t[1] for t in targets if t[0] == region][0]
                    # Smooth ramp: transport increases as source infection grows
                    time_factor = min(1, t / max(delay_info, 1))
                    viral_flux = V_src * VIRAL_TRANSPORT_RATE * time_factor
                    dd[2] += viral_flux

            dydt.extend(dd)

        return dydt

    sol = solve_ivp(ode_system, (0, total_hrs), y0, t_eval=t_eval,
                    method='RK45', rtol=1e-3, atol=1e-6)

    results = {}
    total_survival = 0
    total_neurons = 0
    all_cleared = True

    for i, region in enumerate(REGION_ORDER):
        info = REGIONS[region]
        N0 = info['neurons']
        idx = slice(i * 5, i * 5 + 5)
        S = sol.y[idx][0]
        I = sol.y[idx][1]
        V = sol.y[idx][2]
        E = sol.y[idx][3]
        D = sol.y[idx][4]

        final_S = S[-1]
        survival_frac = final_S / N0
        total_survival += final_S
        total_neurons += N0

        # Clearance = virus + infected cells extinct with no rebound in the tail.
        tail = slice(int((1 - REBOUND_WINDOW) * len(S)), None)
        cleared = bool(V[-1] < EXTINCTION_THRESHOLD and I[-1] < EXTINCTION_THRESHOLD
                       and V[tail].max() < EXTINCTION_THRESHOLD
                       and I[tail].max() < EXTINCTION_THRESHOLD)

        results[region] = {
            'S': S, 'I': I, 'V': V, 'E': E, 'D': D,
            'survival_fraction': survival_frac,
            'neurons_saved': final_S,
            'peak_virus': V.max(),
            'peak_virus_time_hrs': t_eval[V.argmax()],
            'final_virus': float(V[-1]),
            'final_infected': float(I[-1]),
            # 'cured' kept for backward compatibility; it now means viral CLEARANCE.
            'cured': cleared,
            'cleared': cleared,
        }
        if not cleared:
            all_cleared = False

    results['_global'] = {
        't': t_eval,
        'total_neurons_saved': total_survival,
        'total_neurons': total_neurons,
        'overall_survival': total_survival / total_neurons,
        'all_regions_cleared': all_cleared,
        'all_regions_cured': all_cleared,  # backwards-compatible alias
        'treatment_delay_hrs': treatment_delay_hrs,
        'dose_interval_hrs': dose_interval_hrs,
        'drug_cmax': drug_cmax,
        'drug_trough': drug_trough,
    }

    return results


def print_report(results):
    """Print spatial brain model results."""
    g = results['_global']
    print(f"\n{'='*65}")
    print(f"  SPATIAL BRAIN MODEL — RABIES SPREAD")
    print(f"{'='*65}")
    print(f"  Total neurons:      {g['total_neurons']:,.0f}")
    print(f"  Neurons preserved:  {g['overall_survival']*100:.1f}%")
    print(f"  All regions cleared:{' YES' if g['all_regions_cleared'] else ' NO'} "
          f"(virus + infected cells extinct in every region)")
    print(f"  Treatment delay:    {g['treatment_delay_hrs']}h")
    print(f"  Dose interval:      {g['dose_interval_hrs']}h")
    print(f"\n  {'Region':<18} {'Neurons':>9} {'Survival':>9} {'Peak V':>9} {'FinalV':>9} {'Cleared':>8}")
    print(f"  {'-'*62}")

    for region in REGION_ORDER:
        r = results[region]
        if r is None:
            continue
        info = REGIONS[region]
        pct = r['survival_fraction'] * 100
        peak = r['peak_virus']
        cleared = 'YES' if r.get('cleared') else 'NO'
        print(f"  {region:<18} {int(info['neurons']):>9,} {pct:>7.2f}% {peak:>8.0f} "
              f"{r['final_virus']:>8.2g} {cleared:>8}")

    print(f"{'='*65}")
    return g['overall_survival'], g['all_regions_cleared']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmax', type=float, default=73.6)
    parser.add_argument('--trough', type=float, default=3.17)
    parser.add_argument('--delay', type=float, default=48)
    parser.add_argument('--interval', type=float, default=24)
    args = parser.parse_args()

    results = simulate_spatial_rabies(
        drug_cmax=args.cmax,
        drug_trough=args.trough,
        treatment_delay_hrs=args.delay,
        dose_interval_hrs=args.interval,
    )

    print_report(results)
